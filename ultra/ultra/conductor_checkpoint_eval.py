"""Per-checkpoint conductor evaluation for the pool-D GRPO campaign.

Runs at every 10th trainer step (checkpoints land every 5; evals at 10, 20,
30, ...). The typed-contract gate's aggregate counts saturated before training
began (40/40 parse, 40/40 legal — measured at step 10), so the metrics that
carry signal here are TOPOLOGY SHAPE, computed from the gate's per-case
`predicted` plans:

- steps/plan            (expected refs: 3.56; parent 4.44 -> step-10 4.28)
- independent fraction  (no-access positions; refs 27%; parent 55% -> 60%)
- access-edges/plan     (refs 5.06; parent 3.83 -> 3.28)
- |steps - expected|    (parent 0.89 -> 0.83)

Protocol per eval (all zero-paid, collection keeps running — the gate is 40
temperature-0 requests, negligible next to 256-rollout collection):

1. Freeze `broadcasts/step_N` to `eval_frozen/stepN` and md5-verify the copy.
   Evaluating the live `fugu-pool-d-policy` id would let the next trainer
   broadcast mutate the model mid-eval.
2. Register the frozen dir on the conductor under a distinct id via
   `/v1/load_lora_adapter` (grpo-infer admin route; stock vLLM lacks it).
3. Run the typed gate against that id; write the report next to the frozen
   adapter.
4. Compute shape metrics, compare against the parent report and every prior
   checkpoint row, and append to `eval_frozen/shape_ledger.jsonl`.

Usage:
    .venv/bin/python -m ultra.conductor_checkpoint_eval --step 20
    .venv/bin/python -m ultra.conductor_checkpoint_eval --report <gate.json>  # metrics only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "output/fugu_ultra_pool_d"
FROZEN = RUN / "eval_frozen"
LEDGER = FROZEN / "shape_ledger.jsonl"
PARENT_REPORT = FROZEN / "gate_parent_r4base.json"
CONDUCTOR = "http://127.0.0.1:8011"
GATE = REPO / "scratchpad/gate_fugu_ornith_typed_base_v1.py"


def shape_metrics(report_path: Path) -> dict:
    """Topology-shape metrics from a typed-gate report's per-case plans."""
    results = json.loads(report_path.read_text())["results"]
    n_steps = ind = edges = plans = 0
    abs_err = exact = 0.0
    for r in results:
        p = r.get("predicted") or {}
        steps = p.get("steps") or []
        if p.get("action") != "replan" or not steps:
            continue
        plans += 1
        n_steps += len(steps)
        ind += sum(1 for s in steps if not s.get("access"))
        edges += sum(len(s.get("access") or []) for s in steps)
        exp = len((r.get("expected") or {}).get("steps") or [])
        if exp:
            abs_err += abs(len(steps) - exp)
            exact += len(steps) == exp
    if not plans:
        return {"plans": 0}
    return {
        "plans": plans,
        "steps_per_plan": round(n_steps / plans, 3),
        "independent_frac": round(ind / n_steps, 3) if n_steps else None,
        "access_edges_per_plan": round(edges / plans, 3),
        "abs_size_err": round(abs_err / plans, 3),
        "exact_size_match": int(exact),
        **size_adaptivity(results),
        # aggregate gate counts, for the record (expected to stay saturated)
        "counts": json.loads(report_path.read_text())["counts"],
    }


def size_adaptivity(results: list[dict]) -> dict:
    """Does predicted plan size TRACK per-task expected size? (Conductor paper
    §4: the trained conductor 'allocates more compute to harder problems'.)

    The transport samples carry no env id, so per-lane sizing is not
    measurable from rollouts; the gate's frozen cases DO vary in expected
    topology size, which gives task-conditional sizing directly:
    - size_corr: Pearson r between predicted and expected step counts
    - steps_small/steps_large: mean predicted size on cases whose expected
      plan is <=3 steps vs >=4 steps
    - size_delta: steps_large - steps_small (positive = paper-style
      difficulty-adaptive sizing; ~0 = one-size-fits-all)
    """
    pairs = []
    for r in results:
        p_ = r.get("predicted") or {}
        e_ = r.get("expected") or {}
        if p_.get("action") != "replan" or not p_.get("steps"):
            continue
        exp = len(e_.get("steps") or [])
        if exp:
            pairs.append((len(p_["steps"]), exp))
    if len(pairs) < 4:
        return {}
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    n = len(pairs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in pairs)
    vx = sum((a - mx) ** 2 for a in xs) ** 0.5
    vy = sum((b - my) ** 2 for b in ys) ** 0.5
    corr = cov / (vx * vy) if vx and vy else 0.0
    small = [a for a, b in pairs if b <= 3]
    large = [a for a, b in pairs if b >= 4]
    out = {"size_corr": round(corr, 3), "size_pairs": n}
    if small and large:
        out["steps_small"] = round(sum(small) / len(small), 3)
        out["steps_large"] = round(sum(large) / len(large), 3)
        out["size_delta"] = round(out["steps_large"] - out["steps_small"], 3)
    return out


def reward_diversity(last_n: int = 10) -> dict:
    """Within-group reward diversity from the campaign's own rollout batches.

    THE exploration-health signal (measured 2026-08-01): group reward std held
    0.26 -> 0.28 across batches 0-32 even while topology templated, refuting
    the exploration-collapse reading. Failure signatures that WOULD warrant a
    correction (e.g. enabling the QeRL noise scheduler): mean_group_std below
    ~0.15 sustained, zero-variance groups climbing, or reward slope negative.
    """
    import msgspec

    from surogate.grpo.transport.types import TrainingBatch

    dirs = sorted(
        (RUN / "run_default/rollouts").glob("step_*/rollouts.bin"),
        key=lambda x: int(x.parent.name.split("_")[1]),
    )[-last_n:]
    stds, zero, means = [], 0, []
    for path in dirs:
        try:
            batch = msgspec.msgpack.decode(path.read_bytes(), type=TrainingBatch)
        except Exception:  # noqa: BLE001
            continue
        groups: dict[str, list[float]] = {}
        for e in batch.examples:
            if e.reward is None:
                continue
            groups.setdefault(hashlib.md5(bytes(str(e.prompt_ids), "utf8")).hexdigest(), []).append(e.reward)
        big = [v for v in groups.values() if len(v) >= 32]
        for v in big:
            m = sum(v) / len(v)
            sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
            stds.append(sd)
            zero += sd < 1e-6
            means.append(m)
    if not stds:
        return {}
    return {
        "diversity_batches": last_n,
        "mean_group_std": round(sum(stds) / len(stds), 4),
        "zero_var_groups": zero,
        "mean_reward": round(sum(means) / len(means), 4),
    }


def anchored_reward() -> dict:
    """Per-task learning curve from questions the buffer served MORE THAN ONCE.

    Raw batch reward is composition-confounded: difficulty recycling makes the
    mix harder over time (first-encounter reward 0.515 -> 0.454 by batch 34+,
    measured 2026-08-01), so a flat batch mean can hide gains. This tracks the
    same question early vs late (>=5 batches apart). Sparse at first (13
    repeats by batch 51) and grows ~linearly with the run.
    """
    import msgspec

    from surogate.grpo.transport.types import TrainingBatch

    seen: dict[str, list[tuple[int, float]]] = {}
    for path in sorted(
        (RUN / "run_default/rollouts").glob("step_*/rollouts.bin"),
        key=lambda x: int(x.parent.name.split("_")[1]),
    ):
        n = int(path.parent.name.split("_")[1])
        try:
            batch = msgspec.msgpack.decode(path.read_bytes(), type=TrainingBatch)
        except Exception:  # noqa: BLE001
            continue
        groups: dict[str, list[float]] = {}
        for e in batch.examples:
            if e.reward is None:
                continue
            groups.setdefault(hashlib.md5(bytes(str(e.prompt_ids), "utf8")).hexdigest(), []).append(e.reward)
        for h, v in groups.items():
            if len(v) >= 32:
                seen.setdefault(h, []).append((n, sum(v) / len(v)))
    rep = [occ for occ in seen.values() if len(occ) >= 2 and occ[-1][0] - occ[0][0] >= 5]
    if not rep:
        return {"anchored_repeats": 0}
    deltas = [occ[-1][1] - occ[0][1] for occ in rep]
    return {
        "anchored_repeats": len(rep),
        "anchored_up": sum(1 for d in deltas if d > 0.02),
        "anchored_down": sum(1 for d in deltas if d < -0.02),
        "anchored_mean_delta": round(sum(deltas) / len(deltas), 4),
    }


_STOP = frozenset("the a an and or of to in for with on at by is are be as it this that".split())


def plan_diversity(report_path: Path) -> dict:
    """Split multi-step plans into DIVERSE vs DUPLICATE-HEAVY.

    The env's `_redundancy_penalty` taxes near-duplicate subtasks (token
    Jaccard >= 0.75, 0.1/dup, cap 0.3) — measured 2026-08-02 to be the force
    shrinking topologies (within-group big-vs-small deltas -0.13..-0.26 match
    the penalty arithmetic). The escape route inside the objective is
    diverse-phrasing decompositions; this metric watches whether the policy
    finds it: diverse_multistep rising = learning untaxed trees;
    all mass at sizes 1-2 with reward plateau = penalty rescope discussion.
    """
    import re as _re

    results = json.loads(report_path.read_text())["results"]
    small = diverse = dup = 0
    for r in results:
        raw = r.get("raw") or ""
        subs = _re.findall(r'"subtask"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        if not subs:
            continue
        if len(subs) <= 2:
            small += 1
            continue
        sets = []
        for t in subs:
            toks = frozenset(_re.findall(r"[a-z]+", t.lower())) - _STOP
            if toks:
                sets.append(toks)
        redundant = False
        for i in range(1, len(sets)):
            for j in range(i):
                u = sets[i] | sets[j]
                if u and len(sets[i] & sets[j]) / len(u) >= 0.75:
                    redundant = True
                    break
            if redundant:
                break
        if redundant:
            dup += 1
        else:
            diverse += 1
    return {"plans_small": small, "diverse_multistep": diverse, "dup_multistep": dup}


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def freeze_adapter(step: int) -> Path:
    src = RUN / f"run_default/broadcasts/step_{step}"
    if not (src / "STABLE").exists():
        raise SystemExit(f"broadcast step_{step} has no STABLE marker — refusing to eval a partial adapter")
    dst = FROZEN / f"step{step}"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(src / name, dst / name)
        if _md5(src / name) != _md5(dst / name):
            raise SystemExit(f"copy verification failed for {name}")
    return dst


def register_adapter(step: int, path: Path) -> str:
    name = f"fugu-pool-d-step{step}"
    # Unload first: after the 2026-08-02 rewind, step numbers repeat across
    # eras, and vLLM keeps the OLD adapter under an already-registered name.
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"{CONDUCTOR}/v1/unload_lora_adapter",
            data=json.dumps({"lora_name": name}).encode(),
            headers={"Content-Type": "application/json"},
        ), timeout=60).read()
    except Exception:  # noqa: BLE001 — not previously registered is fine
        pass
    req = urllib.request.Request(
        f"{CONDUCTOR}/v1/load_lora_adapter",
        data=json.dumps({"lora_name": name, "lora_path": str(path)}).encode(),
        headers={"Content-Type": "application/json"},
    )
    body = urllib.request.urlopen(req, timeout=120).read().decode()
    if "success" not in body.lower():
        raise SystemExit(f"adapter registration failed: {body[:200]}")
    return name


def run_gate(model_id: str, report: Path) -> None:
    env = {
        "FUGU_GATE_BASE_URL": f"{CONDUCTOR}/v1",
        "FUGU_GATE_MODEL": model_id,
        "FUGU_GATE_REPORT": str(report),
        "PYTHONPATH": f"{REPO}:{REPO}/director:{REPO}/ultra",
        "PATH": "/usr/bin:/bin",
        "HOME": str(Path.home()),
    }
    r = subprocess.run(
        [str(REPO / "director/.venv/bin/python"), str(GATE)],
        env=env, capture_output=True, text=True, timeout=1800,
    )
    if r.returncode != 0 or not report.exists():
        raise SystemExit(f"gate failed rc={r.returncode}\n{r.stderr[-1500:]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, help="trainer checkpoint step to evaluate")
    ap.add_argument("--report", type=Path, help="only compute shape metrics for an existing report")
    ap.add_argument("--no-probe", action="store_true", help="skip the real-task lane probe")
    args = ap.parse_args()

    if args.report:
        print(json.dumps(shape_metrics(args.report), indent=2))
        return
    if args.step is None:
        ap.error("--step or --report required")

    frozen = freeze_adapter(args.step)
    model_id = register_adapter(args.step, frozen)
    report = FROZEN / f"gate_candidate_step{args.step}.json"
    run_gate(model_id, report)

    # Real-task probe on the frozen id (tau2/crm/tool/repo + heldout math/code,
    # paired vs probe_parent_r4base.json — same seed, same tasks, same pool).
    probe_row: dict = {}
    if not args.no_probe:
        from ultra.heldout_probe import run as run_probe

        ps = run_probe(model_id, FROZEN / f"probe_step{args.step}.json")
        probe_row = {"probe_mean": ps.get("mean_reward"),
                     "probe_lanes": {k: v.get("mean") for k, v in ps.get("lanes", {}).items()}}

    row = {"step": args.step, "model_id": model_id, **shape_metrics(report), **plan_diversity(report), **reward_diversity(), **anchored_reward(), **probe_row}
    parent = shape_metrics(PARENT_REPORT) if PARENT_REPORT.exists() else None

    with LEDGER.open("a") as fh:
        fh.write(json.dumps(row) + "\n")

    print(json.dumps({"candidate": row, "parent": parent}, indent=2))
    if LEDGER.exists():
        print("\n=== ledger so far ===")
        for line in LEDGER.read_text().splitlines():
            d = json.loads(line)
            print(f"step {d['step']:>3}: steps/plan {d.get('steps_per_plan')} | "
                  f"indep {d.get('independent_frac')} | edges {d.get('access_edges_per_plan')} | "
                  f"|err| {d.get('abs_size_err')} | exact {d.get('exact_size_match')}")


if __name__ == "__main__":
    main()
