"""Faithful SWE-Bench via the shipped harness (mini-swe-agent) with per-step routing.

Fugu presents as a single model; mini-swe-agent calls that model each step. So we make a
``FuguModel`` (subclass of mini-swe-agent's OpenRouterModel) whose ``query()`` runs our
SelectionRouter over the running transcript, picks a worker, sets the model_name to that
worker's slug, and delegates to the real harness call. Per-step routing happens inside the
actual benchmark harness => faithful resolve rates + the Fugu mechanism.

Solo baselines: pass ``allowed={worker_id}`` so the router is forced to one worker (gives
the proper per-worker SWE-Bench resolve rate through the real harness).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.models.openrouter_model import OpenRouterModel
from minisweagent.run.benchmarks.swebench import get_sb_environment

from ..fugu.inference import select_worker
from ..shared.providers import openrouter_attribution_headers
from .swebench_env import _instance_image  # reused for grading image resolution


def _render(messages: list[dict]) -> str:
    """Raw role:content surface form for the router (matches our featurizer's expectation)."""
    parts = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, list):  # multimodal -> keep the text spans
            c = " ".join(seg.get("text", "") for seg in c if isinstance(seg, dict))
        parts.append(f"{m.get('role','')}: {c}")
    return "\n".join(parts)


class FuguModel(OpenRouterModel):
    """mini-swe-agent Model that routes each query to a worker via the SelectionRouter."""

    _gpu_lock = threading.Lock()  # serialize the 0.6B featurize across parallel rollouts

    def __init__(self, router, worker_slugs: dict[str, str], allowed=None, **kwargs):
        # max reasoning + cheapest provider, faithful to our Fugu settings
        mk = {"reasoning": {"effort": "high"}, "provider": {"sort": "price"},
              "temperature": 0.0, "max_tokens": 32768,
              "extra_headers": openrouter_attribution_headers()}  # OpenRouter app attribution
        first = next(iter(worker_slugs.values()))
        super().__init__(model_name=first, model_kwargs=mk, **kwargs)
        self.router = router
        self.worker_slugs = worker_slugs
        self.allowed = allowed
        self.worker_sequence: list[str] = []
        self.total_cost = 0.0

    def _calculate_cost(self, response) -> dict[str, float]:
        # Cheapest-provider routing sometimes reports top-level cost=0 with the real cost in
        # cost_details.upstream_inference_cost. Fall back to it (and never raise) so cost_limit
        # still binds and routing to cheap providers doesn't crash the run.
        usage = response.get("usage", {}) or {}
        cost = usage.get("cost", 0.0) or 0.0
        if cost <= 0.0:
            cost = float((usage.get("cost_details") or {}).get("upstream_inference_cost", 0.0) or 0.0)
        return {"cost": cost}

    def query(self, messages: list[dict], **kwargs) -> dict:
        with FuguModel._gpu_lock:  # only one GPU featurize at a time
            wid = select_worker(self.router, _render(messages), allowed=self.allowed)
        self.worker_sequence.append(wid)
        self.config.model_name = self.worker_slugs[wid]
        msg = super().query(messages, **kwargs)
        self.total_cost += float(msg.get("extra", {}).get("cost", 0.0) or 0.0)
        return msg


def grade(instance: dict, patch: str, dataset: str, run_id: str = "director_mini") -> float:
    """Grade a patch with the official SWE-Bench harness (resolved => 1.0)."""
    if not patch or not patch.strip():
        return 0.0
    iid = instance["instance_id"]
    with tempfile.TemporaryDirectory() as td:
        pred = os.path.join(td, "preds.jsonl")
        with open(pred, "w") as f:
            f.write(json.dumps({"instance_id": iid, "model_name_or_path": "director",
                                "model_patch": patch}) + "\n")
        subprocess.run(
            [sys.executable, "-m", "swebench.harness.run_evaluation", "--dataset_name", dataset,
             "--predictions_path", pred, "--max_workers", "1", "--run_id", run_id,
             "--instance_ids", iid, "--cache_level", "env"],
            capture_output=True, text=True, timeout=1800,
        )
        report = f"director.{run_id}.json"
        if not os.path.exists(report):
            return 0.0
        data = json.loads(open(report).read())
        try:
            os.remove(report)
        except OSError:
            pass
        return 1.0 if iid in set(data.get("resolved_ids", [])) else 0.0


def run_instance(router, instance: dict, worker_slugs: dict[str, str], *, allowed=None,
                 dataset: str = "princeton-nlp/SWE-bench_Verified", step_limit: int = 0,
                 cost_limit: float = 1.0, do_grade: bool = True) -> dict:
    """Run mini-swe-agent on one instance with per-step routing; return reward + trace.

    step_limit=0 => unlimited steps (the harness default); the run is bounded by cost_limit
    (cheap with open models, so the agent has room to actually edit + submit a patch)."""
    config = get_config_from_spec(str(builtin_config_dir / "benchmarks" / "swebench.yaml"))
    env = get_sb_environment(config, instance)
    model = FuguModel(router, worker_slugs, allowed=allowed)
    acfg = dict(config.get("agent", {}))
    acfg.pop("agent_class", None)
    acfg.update(cost_limit=cost_limit, step_limit=step_limit)
    agent = DefaultAgent(model, env, **acfg)
    info = agent.run(instance["problem_statement"])
    patch = info.get("submission", "") or ""
    reward = grade(instance, patch, dataset) if do_grade else -1.0
    return {
        "instance_id": instance["instance_id"],
        "reward": reward,
        "exit_status": info.get("exit_status"),
        "worker_sequence": model.worker_sequence,
        "cost": model.total_cost,
        "patch": patch,
        "patch_len": len(patch),
    }


def grade_swesmith(instance: dict, patch: str, run_id: str = "director_sm") -> float:
    """Grade a patch with SWE-smith's OWN harness (it ships the per-repo test profiles + images that
    swebench's spec-map lacks). Pulls a FRESH container from instance['image_name'], applies the model
    patch, runs FAIL_TO_PASS+PASS_TO_PASS. Returns 1.0 iff resolved."""
    if not patch or not patch.strip():
        return 0.0
    import glob
    import shutil

    from swebench.harness.constants import RUN_EVALUATION_LOG_DIR

    iid = instance["instance_id"]
    with tempfile.TemporaryDirectory() as td:
        dpath = os.path.join(td, "ds.json")  # swesmith eval needs the FULL instance (tests + image)
        with open(dpath, "w") as f:
            json.dump([instance], f)
        pred = os.path.join(td, "preds.jsonl")
        with open(pred, "w") as f:
            f.write(json.dumps({"instance_id": iid, "model_name_or_path": "director",
                                "model_patch": patch}) + "\n")
        subprocess.run(
            [sys.executable, "-m", "swesmith.harness.eval", "-d", dpath, "-p", pred,
             "--run_id", run_id, "-i", iid, "-w", "1"],
            capture_output=True, text=True, timeout=1800,
        )
        out_dir = RUN_EVALUATION_LOG_DIR / run_id / iid
        resolved = False
        for m in glob.glob(str(out_dir / "*.json")):  # report filename is robust to LOG_REPORT name
            try:
                d = json.loads(open(m).read())
                if "resolved" in d:
                    resolved = bool(d["resolved"])
                    break
            except (OSError, json.JSONDecodeError):
                pass
        shutil.rmtree(RUN_EVALUATION_LOG_DIR / run_id, ignore_errors=True)
        return 1.0 if resolved else 0.0


def run_swesmith_instance(router, instance: dict, worker_slugs: dict[str, str], *, allowed=None,
                          step_limit: int = 0, cost_limit: float = 1.0, do_grade: bool = True) -> dict:
    """mini-swe-agent on a SWE-smith instance (same /testbed + root convention as SWE-bench), graded
    by SWE-smith's harness. The image ships in the instance ('image_name') -> no swebench spec
    needed. Per-step routing happens inside mini-swe-agent via FuguModel."""
    inst = dict(instance)
    inst["docker_image"] = inst.get("image_name") or inst.get("docker_image")
    config = get_config_from_spec(str(builtin_config_dir / "benchmarks" / "swebench.yaml"))
    env = get_sb_environment(config, inst)
    # SWE-smith images are per-REPO (shared across all bug branches) and default to a CLEAN main.
    # The bug for this instance lives on the branch named by instance_id -> check it out, or the
    # agent rolls out on clean code (no bug -> empty patch -> 0). (conda env `testbed` is already
    # auto-activated because the env interpreter is a login shell, `bash -lc`.)
    iid = inst["instance_id"]
    co = env.execute({"command": f"git checkout {iid}"}, cwd="/testbed")
    if co.get("returncode", 1) != 0:
        raise RuntimeError(f"git checkout {iid} failed: {co.get('output','')[:300]}")
    model = FuguModel(router, worker_slugs, allowed=allowed)
    acfg = dict(config.get("agent", {}))
    acfg.pop("agent_class", None)
    acfg.update(cost_limit=cost_limit, step_limit=step_limit)
    agent = DefaultAgent(model, env, **acfg)
    info = agent.run(inst["problem_statement"])
    patch = info.get("submission", "") or ""
    reward = grade_swesmith(inst, patch) if do_grade else -1.0
    return {
        "instance_id": inst["instance_id"], "reward": reward,
        "exit_status": info.get("exit_status"), "worker_sequence": model.worker_sequence,
        "cost": model.total_cost, "patch": patch, "patch_len": len(patch),
    }


def grade_batch(patches_by_iid: dict[str, str], dataset: str, run_id: str, max_workers: int = 4) -> set[str]:
    """Grade many (instance_id -> patch) in ONE harness invocation; return resolved ids."""
    preds = {iid: {"instance_id": iid, "model_name_or_path": "director", "model_patch": p}
             for iid, p in patches_by_iid.items() if p and p.strip()}
    if not preds:
        return set()
    with tempfile.TemporaryDirectory() as td:
        pred = os.path.join(td, "preds.jsonl")
        with open(pred, "w") as f:
            for v in preds.values():
                f.write(json.dumps(v) + "\n")
        subprocess.run(
            [sys.executable, "-m", "swebench.harness.run_evaluation", "--dataset_name", dataset,
             "--predictions_path", pred, "--max_workers", str(max_workers), "--run_id", run_id,
             "--instance_ids", *preds.keys(), "--cache_level", "env"],
            capture_output=True, text=True, timeout=3600,
        )
        report = f"director.{run_id}.json"
        if not os.path.exists(report):
            return set()
        data = json.loads(open(report).read())
        try:
            os.remove(report)
        except OSError:
            pass
        return set(data.get("resolved_ids", []))
