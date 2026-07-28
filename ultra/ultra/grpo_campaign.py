"""r5 GRPO campaign driver: curated rollout collection feeding the trainer.

Composes the shipped pieces into the run MISSION.md specifies — reused hard
mix as the candidate pool, probe-then-train curation, online attrition, and
exact-token batches — with the trainer injected so the driver is
offline-testable and the paid components are wired only at launch.

    candidates (hard-mix taskspecs, prior-ordered)
      -> ProbeFilter (m=3 under the CURRENT system; journaled)
      -> choose_group_size (from the live keep-rate)
      -> step loop: sample questions -> collect groups at G
                    -> AttritionTracker -> build_conductor_batch -> trainer

Every retention decision is outcome-based (reward variance only); the
stage2-era difficulty priors are used ONLY to order probing so the spend
stops early, never to retain or drop a question.

PROMPT-CONVENTION CONTRACT. `Candidate.question` is the TASK PAYLOAD — the
flattened [system]/[user] task text a conductor plans over (and the key the
stage2 priors are stored under). It is NOT the conductor's own prompt. The
8B stage2 run wrapped this payload in the Conductor paper's system prompt;
the r5 (27B) campaign wraps it in the 27B's native typed-contract template.
The injected `collector.sampler` and `tokenize_prompt` MUST both apply that
same serving template, byte-identical, or the training tokens will not be
the behavior policy's tokens (the exact-token rule in
grpo_conductor_batch).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from ultra.grpo_conductor_batch import build_conductor_batch
from ultra.grpo_rollout_collector import RolloutCollector
from ultra.question_curation import (
    DEFAULT_ATTRITION_AFTER,
    DEFAULT_PROBE_SIZE,
    AttritionTracker,
    ProbeFilter,
)

# hard_mix_all is deliberately ABSENT: it is a subset of the component files
# and contributes zero unique candidates (pre-flight review, 2026-07-27).
MIX_FILES = (
    "hard_mix_math_taskspecs.jsonl",
    "hard_mix_rlpr_taskspecs.jsonl",
    "hard_mix_code_taskspecs.jsonl",
    "hard_mix_repair_taskspecs.jsonl",
    # Office lanes with registry graders curate through the probe exactly
    # like the others; each file is a deterministic sample of a much larger
    # train split (see the write_probe_sample in each exporter).
    "bird_probe_candidates_taskspecs.jsonl",       # sql_exec
    "finance_probe_candidates_taskspecs.jsonl",    # finance_numeric
    "dabstep_probe_candidates_taskspecs.jsonl",    # dabstep_exec
)

# Golds above this size are excluded outright: 8 LCB questions carry test
# payloads up to 110 MB — they bloat every journal write and stall grading
# on the serving host for no training value the other 200+ code questions
# don't already provide.
MAX_GOLD_BYTES = 1_000_000


@dataclass(frozen=True)
class Candidate:
    question: str          # rendered exactly as the conductor is prompted
    gold: Any
    grader_type: str
    task_id: str
    prior_uninformative: float | None = None  # stage2 mine, ordering only


def render_question(messages: list[dict[str, str]]) -> str:
    """Flatten taskspec messages to the conductor's question text.

    Byte-compatible with the stage2 mine's `task` field ([system]/[user]
    blocks), so difficulty priors key onto candidates exactly.
    """
    return "\n\n".join(
        f"[{m['role']}]\n{m['content']}" for m in messages
    )


def load_mix_candidates(
    manifest_dir: Path,
    priors_path: Path | None = None,
    include_code: bool = True,
    max_gold_bytes: int = MAX_GOLD_BYTES,
) -> list[Candidate]:
    """Load, dedup, and prior-order the hard-mix candidate pool.

    Dedup is by question text AND by problem identity: repair_* tasks are
    the same underlying problems as their code counterparts (40/40 exact
    gold matches), so `repair_taco__1` and `taco__1` count once.

    Order: prior-informative questions first (ascending uninformative_share
    — the stage2 mine's 103 lead), then unprobed candidates in file order.
    Never-informative-in-stage2 questions go last, NOT dropped: the prior is
    stale w.r.t. the current system, so they still get their m=3 probe.
    """
    priors: dict[str, float] = {}
    if priors_path and priors_path.exists():
        for line in priors_path.read_text().split("\n"):
            if line.strip():
                row = json.loads(line)
                priors[row["task"].strip()] = float(row["uninformative_share"])

    seen: set[str] = set()
    seen_problems: set[str] = set()
    candidates: list[Candidate] = []
    for name in MIX_FILES:
        path = manifest_dir / name
        if not path.exists():
            continue
        for line in path.read_text().split("\n"):
            if not line.strip():
                continue
            spec = json.loads(line)
            grader_type = spec["grader"]["type"]
            if not include_code and grader_type == "code_exec_stdio":
                continue
            gold = spec["grader"]["expected_answer"]
            if len(json.dumps(gold)) > max_gold_bytes:
                continue  # giant test payloads: excluded (see MAX_GOLD_BYTES)
            problem_id = str(spec.get("task_id", "")).removeprefix("repair_")
            if problem_id and problem_id in seen_problems:
                continue
            question = render_question(spec["input"]["messages"])
            key = question.strip()
            if key in seen:
                continue
            seen.add(key)
            if problem_id:
                seen_problems.add(problem_id)
            prior = None
            for p_task, p_share in priors.items():
                if key.startswith(p_task[:200]) or p_task.startswith(key[:200]):
                    prior = p_share
                    break
            candidates.append(Candidate(
                question=question,
                gold=spec["grader"]["expected_answer"],
                grader_type=grader_type,
                task_id=spec["task_id"],
                prior_uninformative=prior,
            ))

    def order(item: tuple[int, Candidate]):
        index, candidate = item
        if candidate.prior_uninformative is None:
            return (1, 0.0, index)          # unprobed: after known-informative
        return (0, candidate.prior_uninformative, index)

    return [c for _, c in sorted(enumerate(candidates), key=order)]


def interleave_by_type(candidates: list[Candidate]) -> list[Candidate]:
    """Round-robin candidates across grader types, preserving in-type order.

    The file-ordered pool probes one type at a time, which mattered when a
    retention target stopped the probe early (cheapest-informative first).
    When the whole pool is probed anyway, interleaving is strictly better:
    types with LOCAL grading cost (dabstep script execution) overlap with
    provider-latency-bound types instead of stacking up at the end, and
    per-type keep-rates become visible early instead of after hours.
    """
    by_type: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_type.setdefault(candidate.grader_type, []).append(candidate)
    queues = [list(reversed(group)) for group in by_type.values()]
    out: list[Candidate] = []
    while queues:
        for queue in list(queues):
            out.append(queue.pop())
            if not queue:
                queues.remove(queue)
    return out


def choose_group_size(
    keep_rate: float,
    retained: int,
    questions_per_step: int,
    steps: int,
) -> int:
    """Pick training G from the live probe keep-rate (MISSION: G=16-32).

    The curated-pool yield curve is nearly flat past G=16 (grad/rollout
    0.92 -> 0.97 from G=8 to G=32), so more distinct questions beats deeper
    groups when retention allows it: G=16 when each retained question is
    drawn at most ~4 times over the campaign AND the probe kept a healthy
    share; G=32 when retention is thin and depth is the only place spend
    can go.
    """
    if retained <= 0:
        raise ValueError("no retained questions; cannot size groups")
    draws = questions_per_step * steps
    return 16 if retained >= draws / 4 and keep_rate >= 0.25 else 32


class TrainerStep(Protocol):
    """Deliver one exact-token batch to the trainer; returns step metrics.

    CONTRACT (pre-flight review items 3): `batch.step` is a CONTIGUOUS
    0-based counter over batches actually delivered — campaign steps that
    produce no informative groups consume no number, so the filesystem
    transport's contiguity requirement holds. The adapter MUST raise if the
    trainer stops consuming (ack/health check): the campaign treats an
    exception as "halt spend now" and never swallows it.
    """

    def __call__(self, batch: Any, stats: Any) -> dict[str, Any]: ...


@dataclass
class CampaignConfig:
    steps: int = 200
    questions_per_step: int = 4
    group_size: int | None = None       # None: choose_group_size at launch
    probe_size: int = DEFAULT_PROBE_SIZE
    retention_target: int | None = None  # probe early-stop (None: probe all)
    probe_concurrency: int = 6           # candidates probed in parallel
    attrition_after: int = DEFAULT_ATTRITION_AFTER
    temperature: float = 1.0
    seed: int = 20260727
    checkpoint_every: int = 10          # serving-reload cadence (in SENT batches)


@dataclass
class GrpoCampaign:
    """The r5 training loop. All paid components injected.

    Crash-safe: with a `journal_dir`, campaign state (step, sent-batch
    counter, attrition, RNG) persists after every step and `run()` resumes
    from it — a crash at step 150 re-buys nothing. `on_checkpoint` fires
    every `checkpoint_every` SENT batches (the serving-reload hook); the
    campaign blocks until it returns, so collection never races a reload.
    """

    collector: RolloutCollector
    trainer_step: TrainerStep
    tokenize_prompt: Callable[[str], list[int]]
    tokenize_completion: Callable[[str], list[int]]
    config: CampaignConfig = field(default_factory=CampaignConfig)
    journal_dir: Path | None = None
    on_checkpoint: Callable[[int], None] | None = None
    log: Callable[[str], None] = print

    # ------------------------------------------------------------- state

    def _state_path(self) -> Path | None:
        return (self.journal_dir / "campaign_state.json"
                if self.journal_dir else None)

    def _save_state(self, step: int, sent: int, tracker: AttritionTracker,
                    rng: random.Random, group_size: int) -> None:
        path = self._state_path()
        if path is None:
            return
        state = {
            "next_step": step + 1,
            "sent_batches": sent,
            "group_size": group_size,
            "attrition": tracker.state(),
            "rng": _rng_state_to_json(rng.getstate()),
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(path)

    def _load_state(self) -> dict[str, Any] | None:
        path = self._state_path()
        if path is None or not path.exists():
            return None
        return json.loads(path.read_text())

    # ------------------------------------------------------------- probe

    def probe(self, candidates: list[Candidate]):
        probe = ProbeFilter(
            collector=self.collector,
            probe_size=self.config.probe_size,
            retention_target=self.config.retention_target,
            concurrency=self.config.probe_concurrency,
            journal_path=(self.journal_dir / "probe.jsonl"
                          if self.journal_dir else None),
            log=self.log,
        )
        triples = [(c.question, c.gold, c.grader_type) for c in candidates]
        retained_triples, probe_groups, stats = probe.probe(triples)
        retained_set = {t[0] for t in retained_triples}
        retained = [c for c in candidates if c.question in retained_set]
        return retained, probe_groups, stats

    # --------------------------------------------------------------- run

    def _assert_temperature(self) -> None:
        """The sampler's temperature MUST match what the batch stamps —
        completion_temperatures feeds the trainer's logprob scaling."""
        sampler_temp = getattr(self.collector.sampler, "temperature", None)
        if sampler_temp is None:
            sampler = self.collector.sampler
            sampler_temp = getattr(getattr(sampler, "__self__", None),
                                   "temperature", None)
        if sampler_temp is not None and \
                abs(sampler_temp - self.config.temperature) > 1e-9:
            raise ValueError(
                f"sampler temperature {sampler_temp} != campaign temperature "
                f"{self.config.temperature}; the exact-token batch would "
                f"stamp the wrong temperature")

    def _deliver(self, groups, sent: int) -> tuple[Any, Any]:
        """Build a batch stamped with the SENT counter (contiguous 0-based)."""
        return build_conductor_batch(
            groups, self.tokenize_prompt, self.tokenize_completion,
            step=sent, temperature=self.config.temperature)

    def run(self, candidates: list[Candidate]) -> dict[str, Any]:
        cfg = self.config
        self._assert_temperature()
        retained, probe_groups, probe_stats = self.probe(candidates)
        if not retained:
            raise RuntimeError(
                "probe retained no questions — the candidate pool is "
                "saturated for the current system; do not start training")

        state = self._load_state()
        if state is not None:
            group_size = state["group_size"]
            start_step = state["next_step"]
            sent = state["sent_batches"]
            tracker = AttritionTracker.from_state(
                state["attrition"], attrition_after=cfg.attrition_after)
            rng = random.Random()
            rng.setstate(_rng_state_from_json(state["rng"]))
            self.log(f"[campaign] RESUMED at step {start_step} "
                     f"(sent {sent} batches; {len(tracker.dropped)} attrited)")
        else:
            group_size = cfg.group_size or choose_group_size(
                probe_stats["keep_rate"], len(retained),
                cfg.questions_per_step, cfg.steps)
            start_step = 1
            sent = 0
            tracker = AttritionTracker(attrition_after=cfg.attrition_after)
            rng = random.Random(cfg.seed)
        self.log(f"[campaign] retained {len(retained)} questions "
                 f"(keep rate {probe_stats['keep_rate']:.3f}), G={group_size}")

        history: list[dict[str, Any]] = []
        self.collector.group_size = group_size

        # Recycle probe spend (review item 8): the retained questions'
        # informative probe groups are valid on-policy training groups —
        # they become batch 0 on a fresh start.
        if state is None and probe_groups:
            batch, stats = self._deliver(probe_groups, sent)
            if batch.examples:
                metrics = {"step": 0, "source": "probe-recycle",
                           **stats.as_dict(),
                           **(self.trainer_step(batch, stats) or {})}
                history.append(metrics)
                sent += 1
                self.log(f"[campaign] probe recycle: "
                         f"{stats.rollouts_used} rollouts, "
                         f"{stats.trainable_tokens} tokens as batch 0")

        journal = (self.journal_dir / "train_groups.jsonl"
                   if self.journal_dir else None)
        handle = journal.open("a") if journal else None
        try:
            for step in range(start_step, cfg.steps + 1):
                active = tracker.filter(
                    [(c.question, c.gold, c.grader_type) for c in retained])
                if not active:
                    self.log(f"[campaign] step {step}: every retained "
                             f"question attrited — policy has saturated the "
                             f"pool; stopping early")
                    break
                picks = [active[rng.randrange(len(active))]
                         for _ in range(cfg.questions_per_step)]
                groups = []
                for question, gold, grader_type in picks:
                    group = self.collector.collect_question(
                        question, gold, grader_type)
                    tracker.observe(group)
                    groups.append(group)
                    if handle:
                        journal_row = dict(group)
                        journal_row.pop("gold", None)  # golds can be huge;
                        # candidates re-supply them on resume
                        handle.write(json.dumps(journal_row) + "\n")
                        handle.flush()
                batch, stats = self._deliver(groups, sent)
                metrics: dict[str, Any] = {"step": step,
                                           "active_questions": len(active),
                                           **stats.as_dict()}
                if batch.examples:
                    metrics["batch_step"] = sent
                    metrics.update(self.trainer_step(batch, stats) or {})
                    sent += 1
                    if (self.on_checkpoint is not None
                            and sent % cfg.checkpoint_every == 0):
                        self.log(f"[campaign] checkpoint hook at "
                                 f"{sent} sent batches (serving reload)")
                        self.on_checkpoint(sent)
                else:
                    metrics["skipped"] = "no informative groups this step"
                history.append(metrics)
                self._save_state(step, sent, tracker, rng, group_size)
                self.log(f"[campaign] step {step}/{cfg.steps} "
                         f"groups_used={stats.groups_used}/"
                         f"{stats.groups_in} "
                         f"tokens={stats.trainable_tokens} "
                         f"active={len(active)}")
        finally:
            if handle:
                handle.close()

        loop_steps = [m for m in history if m.get("step", 0) >= 1]
        informative_steps = sum(1 for m in loop_steps if "skipped" not in m)
        return {
            "probe": probe_stats,
            "group_size": group_size,
            "retained": len(retained),
            "steps_run": len(loop_steps),
            "steps_with_gradient": informative_steps,
            "batches_sent": sent,
            "questions_attrited": len(tracker.dropped),
            "history": history,
            "attrition_state": tracker.state(),
        }


def _rng_state_to_json(state) -> list:
    version, internal, gauss = state
    return [version, list(internal), gauss]


def _rng_state_from_json(state) -> tuple:
    version, internal, gauss = state
    return (version, tuple(internal), gauss)
