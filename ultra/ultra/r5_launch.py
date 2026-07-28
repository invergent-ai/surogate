"""r5 launch assembly: wire every tested component into the paid loop.

    vLLM :8011 (merged 27B)  <--reload--  MergedReloadController
         |  sample_plan (guidelines = stage-2 winner, FROZEN)
    TrainedConductor -> RolloutCollector (per-question grader dispatch)
         |  groups                        ^ workers: openrouter, constrained
    GrpoCampaign (probe -> G -> attrition -> exact-token batches, resumable)
         |  TrainingBatch (contiguous 0-based sent counter)
    FileSystemTrainingBatchSender -> surogate GRPOTrainer (train_r5.yaml)

The TrainerStep contract requires halting spend when the trainer stops
consuming: `TrainerHealth` watches the trainer's metrics file and raises
once the sent-batch lag exceeds `max_lag` with no trainer progress for
`stall_timeout_s`. A raise propagates out of GrpoCampaign.run() — no
further rollouts are bought.

Modes:
  --dry-batch   zero-paid pre-spend check (train_r5.yaml checklist item 1):
                sample 2 real plans from :8011, build a synthetic group,
                send it as batch 0, and report the trainer's mismatch_kl —
                at identical weights it must be ~0 (and ratio_clipped ~0).
  --probe-only  run the probe pass only; report keep-rate and chosen G.
  --run         the full campaign (REFUSES while the stage-2 prompt verdict
                is unset — the serving prompt must be frozen first).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainerHealth:
    """Raises when the trainer stops consuming while spend continues.

    Progress signal: the trainer's metrics jsonl (one line per step). Lag =
    batches sent minus last trainer step observed. Small lags are normal
    (async level, packing); a large lag with a stale metrics file means the
    trainer is dead or wedged — every further rollout would be wasted.
    """

    metrics_path: Path
    max_lag: int = 25
    stall_timeout_s: float = 1800.0
    now: Callable[[], float] = time.monotonic
    _last_progress_at: float | None = None
    _last_seen_step: int = -1

    def trainer_step_seen(self) -> int:
        try:
            last = -1
            with self.metrics_path.open() as handle:
                for line in handle:
                    if line.strip():
                        try:
                            last = max(last, int(json.loads(line).get("step", -1)))
                        except (ValueError, TypeError):
                            continue
            return last
        except OSError:
            return -1

    def check(self, batches_sent: int) -> None:
        seen = self.trainer_step_seen()
        now = self.now()
        if seen > self._last_seen_step:
            self._last_seen_step = seen
            self._last_progress_at = now
            return
        if self._last_progress_at is None:
            self._last_progress_at = now
        lag = batches_sent - max(self._last_seen_step, 0)
        stalled_for = now - self._last_progress_at
        if lag > self.max_lag and stalled_for > self.stall_timeout_s:
            raise RuntimeError(
                f"trainer stalled: {batches_sent} batches sent, last trainer "
                f"step {self._last_seen_step}, no progress for "
                f"{stalled_for:.0f}s — halting spend (TrainerStep contract)")


def build_trainer_step(transport_dir: Path, health: TrainerHealth):
    """The production TrainerStep: filesystem transport + stall detection."""
    from surogate.grpo.transport.filesystem import FileSystemTrainingBatchSender

    sender = FileSystemTrainingBatchSender(transport_dir)

    def trainer_step(batch, stats) -> dict[str, Any]:
        sender.send(batch)
        health.check(batch.step + 1)
        return {"sent_step": batch.step,
                "trainer_step_seen": health.trainer_step_seen()}

    return trainer_step


# --------------------------------------------------------------- assembly

FROZEN_GUIDELINES_PATH = ROOT / "director/manifests/fugu_clean_v1/" \
    "grpo_pilot_train/r5_serving_guidelines.json"


def load_frozen_guidelines() -> str | None:
    """The stage-2 winner, frozen by scratchpad/prompt_ab_stage2.py --decide.

    File format: {"arm": "a|b|c", "guidelines": str|null, "decided": str}.
    Missing file = verdict not yet issued = the paid campaign must refuse.
    """
    if not FROZEN_GUIDELINES_PATH.exists():
        return None
    return json.loads(FROZEN_GUIDELINES_PATH.read_text()).get("guidelines")


def build_campaign(args) -> "Any":
    from ultra.grpo_campaign import CampaignConfig, GrpoCampaign
    from ultra.grpo_rollout_collector import RolloutCollector
    from ultra.serving_reload import MergedReloadController
    from ultra.trained_conductor import TrainedConductor
    from ultra.fugu_heavy import FuguHeavyOrchestrator
    from ultra.worker_loop import FunctionCallingWorkerLoop, openrouter_client

    binding = json.loads(
        (ROOT / "director/manifests/fugu_clean_v1/grpo_pilot_train/"
         "current_pool_binding_general.json").read_text())
    binding_map = {"/".join(s["role_prior"]): s["runtime_model"]
                   for s in binding["slots"]}
    roles = [tuple(s["role_prior"]) for s in binding["slots"]]
    role_names = ["/".join(t) for t in roles]

    if not FROZEN_GUIDELINES_PATH.exists():
        raise SystemExit(
            "REFUSING: the stage-2 serving-prompt verdict is not frozen "
            f"({FROZEN_GUIDELINES_PATH} missing). Run prompt_ab_stage2.py "
            "--decide --freeze first; the serving prompt is part of the "
            "exact-token contract.")
    guidelines = load_frozen_guidelines()

    conductor = TrainedConductor(roles=roles, temperature=1.0,
                                 guidelines=guidelines)
    key = _openrouter_key()

    def executor(question: str, steps):
        # TRAINING worker regime (user decision 2026-07-27, after the
        # effort/cap matrix): DEFAULT reasoning effort + max_tokens 16384 +
        # temp 0.2 — "curation as filter". The effort knob cannot make this
        # open-weight pool concise (measured: models fill any budget; low
        # effort still truncates at 6k), so instead the CAP is part of the
        # difficulty filter: the probe retains only questions whose reward
        # variance exists within the 16k budget; questions needing 40k of
        # thinking truncate -> uniform 0.5 -> correctly rejected as
        # untrainable under this regime. Deployment/gates keep full-strength
        # workers; transfer is verified there.
        orch = FuguHeavyOrchestrator(
            binding=binding_map,
            worker_loop=FunctionCallingWorkerLoop(
                client=openrouter_client(key, binding["provider_base"],
                                         max_tokens=16384,
                                         temperature=0.2)),
        )
        orch.conductor = type("Fixed", (), {"plan": lambda self, i, r: steps})()
        return orch.execute(question, env=None)["output"]

    # The conductor's serving window is 8192 tokens and TRAINING truncates
    # its task view at 12k chars (fugu_ultra_pilot._messages_text); the
    # probe must present the same view — DABstep payloads carry a 22k-char
    # manual and overflow the window untruncated (vLLM 400s, probe crash
    # 2026-07-28). Workers still receive the FULL question via the executor.
    def _conductor_view(question: str, max_chars: int = 12000) -> str:
        if len(question) > max_chars:
            return question[:max_chars] + "\n\n[truncated for conductor prompt]"
        return question

    collector = RolloutCollector(
        sampler=lambda q: conductor.sample_plan(_conductor_view(q), role_names),
        executor=executor,
        grader=_refuse_fallback_grader,     # mixed pool: dispatch only
        group_size=8, parallelism=8)

    journal_dir = Path(args.journal_dir)
    journal_dir.mkdir(parents=True, exist_ok=True)
    health = TrainerHealth(metrics_path=Path(args.trainer_metrics))
    reload_controller = MergedReloadController(
        base_model_dir=ROOT / "output/fugu_27b_r4_merged_bf16",
        checkpoints_dir=Path(args.trainer_output),
        merged_root=Path(args.merged_root),
        server_log=journal_dir / "vllm_reloads.log",
    )
    campaign = GrpoCampaign(
        collector=collector,
        trainer_step=build_trainer_step(Path(args.transport_dir), health),
        tokenize_prompt=_no_tokenizer("prompt"),
        tokenize_completion=_no_tokenizer("completion"),
        config=CampaignConfig(
            steps=args.steps, questions_per_step=4,
            retention_target=args.retention_target,
            checkpoint_every=10,
            # Probe wall-clock is provider latency, not GPU: each question is
            # 3 independent 16k-token worker calls, so concurrency only trades
            # wall-clock, never total spend. 6 measured ~10-15 completes/hour,
            # which puts the remaining pool days out.
            probe_concurrency=args.probe_concurrency,
        ),
        journal_dir=journal_dir,
        on_checkpoint=reload_controller.on_checkpoint,
    )
    return campaign


def _refuse_fallback_grader(output: str, gold: Any) -> bool:
    raise RuntimeError(
        "fallback grader reached on the mixed pool — a candidate lost its "
        "grader_type; this is the launch-blocking dispatch bug, halt")


def _no_tokenizer(which: str):
    def fail(_value):
        raise RuntimeError(
            f"{which} tokenizer fallback reached — a rollout lacks served "
            f"token ids (vLLM return_token_ids). Re-tokenized text is NOT "
            f"the behavior policy's tokens; halt and check the serving path")
    return fail


def _openrouter_key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("OPENROUTER_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no OPENROUTER_KEY in .env")


# ---------------------------------------------------------------- dry batch

def dry_batch(args) -> None:
    """Checklist item 1, zero-paid: trainer logprob agreement at identical
    weights. Samples two real plans from :8011 (local GPUs, no workers),
    fabricates a reward split so the group carries advantage, sends it as
    batch 0, then polls the trainer metrics for the step-0 line."""
    from ultra.grpo_conductor_batch import build_conductor_batch
    from ultra.trained_conductor import TrainedConductor

    binding = json.loads(
        (ROOT / "director/manifests/fugu_clean_v1/grpo_pilot_train/"
         "current_pool_binding_general.json").read_text())
    roles = [tuple(s["role_prior"]) for s in binding["slots"]]
    role_names = ["/".join(t) for t in roles]
    conductor = TrainedConductor(roles=roles, temperature=1.0,
                                 guidelines=load_frozen_guidelines())

    question = ("[user]\nWhat is the sum of the first 100 positive "
                "integers? Put the final answer in \\boxed{}.")
    rollouts = []
    for reward in (1.0, 0.5):     # fabricated split: advantage != 0
        _steps, meta = conductor.sample_plan(question, role_names)
        if not meta.get("token_ids"):
            raise SystemExit("serving did not return token_ids — fix "
                             "before any spend")
        rollouts.append({"reward": reward, "raw": meta["raw"],
                         "logprobs": meta["logprobs"],
                         "token_ids": meta["token_ids"],
                         "prompt_token_ids": meta["prompt_token_ids"]})
    group = {"question": question, "gold": "5050", "rollouts": rollouts,
             "rewards": [1.0, 0.5], "informative": True}
    batch, stats = build_conductor_batch(
        [group], _no_tokenizer("prompt"), _no_tokenizer("completion"),
        step=0, temperature=1.0)
    assert stats.rollouts_logprob_fallback == 0, "logprob misalignment"

    from surogate.grpo.transport.filesystem import FileSystemTrainingBatchSender
    FileSystemTrainingBatchSender(Path(args.transport_dir)).send(batch)
    print(f"dry batch 0 sent: {stats.as_dict()}")
    print("now read the trainer metrics for step 0: PASS iff "
          "mismatch_kl ~ 0 and ratio_clipped ~ 0 at identical weights "
          "(grammar-constrained serving vs trainer logprob agreement).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal-dir",
                    default=str(ROOT / "output/fugu_ultra_r5/campaign"))
    ap.add_argument("--transport-dir",
                    default=str(ROOT / "output/fugu_ultra_r5"))
    ap.add_argument("--trainer-output",
                    default=str(ROOT / "output/fugu_ultra_r5"))
    ap.add_argument("--trainer-metrics",
                    default=str(ROOT / "output/fugu_ultra_r5/metrics.jsonl"))
    ap.add_argument("--merged-root",
                    default=str(ROOT / "output/fugu_ultra_r5/merged"))
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--retention-target", type=int, default=350)
    ap.add_argument("--probe-concurrency", type=int, default=16)
    ap.add_argument("--probe-interleave", action="store_true",
                    help="round-robin candidates across grader types (use when "
                         "probing the whole pool; overlaps local grading with "
                         "provider-bound types)")
    ap.add_argument("--dry-batch", action="store_true")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    if args.dry_batch:
        dry_batch(args)
        return

    from ultra.grpo_campaign import interleave_by_type, load_mix_candidates
    candidates = load_mix_candidates(
        ROOT / "director/manifests/fugu_clean_v1/grpo_pilot_train",
        ROOT / "scratchpad/stage2_mined/question_difficulty.jsonl")
    campaign = build_campaign(args)
    if args.probe_only:
        if args.probe_interleave:
            candidates = interleave_by_type(candidates)
        retained, _groups, stats = campaign.probe(candidates)
        print(json.dumps(stats, indent=1))
        return
    if args.run:
        result = campaign.run(candidates)
        print(json.dumps({k: v for k, v in result.items()
                          if k != "history"}, indent=1))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
