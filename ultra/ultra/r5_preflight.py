"""Launch preflight for the r5 interleaved GRPO campaign.

One command that answers "is it safe to spend money on this launch?". Every
check is a fact about the files and processes that will actually be used, not
a restatement of intent: manifests exist and are non-empty, the orch's env
lanes and buffer ratios line up, every live lane's safety manifest is
approved and covers its workers, the trainer and orch agree on sequence
length, and the serving endpoint answers to the exact model name the orch
will request.

Exit code 0 only when every check passes. ``--json`` prints the machine
report; the default output is a readable checklist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = ROOT / "director" / "manifests" / "fugu_clean_v1" / "grpo_pilot_train"


class Check:
    def __init__(self, name: str, ok: bool, detail: str) -> None:
        self.name, self.ok, self.detail = name, ok, detail

    def as_dict(self) -> dict[str, Any]:
        return {"check": self.name, "ok": self.ok, "detail": self.detail}


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_json(path: Path) -> Any:
    """Parse a JSON file, or None if it is missing/unreadable.

    A file that a later launch step still has to produce (the probe's pilot
    config, say) must surface as a FAILED CHECK, never as a traceback — the
    whole point is a report the operator can read before spending money.
    """
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _jsonl_count(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def check_orch_lanes(orch: dict[str, Any]) -> list[Check]:
    """Env lanes, buffer ratios, and group size must be mutually consistent."""
    out: list[Check] = []
    lanes = [e["name"] for e in orch.get("env", [])]
    ratios = orch.get("buffer", {}).get("env_ratios", [])
    out.append(Check(
        "orch.env_ratios_align",
        len(lanes) == len(ratios) and abs(sum(ratios) - 1.0) < 1e-6,
        f"{len(lanes)} lanes, {len(ratios)} ratios, sum={sum(ratios):.4f}",
    ))
    group = orch.get("rollouts_per_example")
    batch = orch.get("batch_size")
    out.append(Check(
        "orch.group_size_divides_batch",
        bool(group) and bool(batch) and batch % group == 0,
        f"batch_size={batch}, rollouts_per_example={group}",
    ))
    advantage = orch.get("advantage") or {}
    if advantage.get("type") == "custom":
        # custom advantage functions receive **kwargs; extra keys at the
        # block level are silently dropped and kwargs=None crashes at step 0
        extras = set(advantage) - {"type", "import_path", "kwargs"}
        ok = isinstance(advantage.get("kwargs"), dict) and not extras
        out.append(Check(
            "orch.custom_advantage_kwargs", ok,
            f"kwargs={'mapping' if isinstance(advantage.get('kwargs'), dict) else advantage.get('kwargs')!r}"
            + (f", stray keys {sorted(extras)}" if extras else ""),
        ))
    return out


def check_worker_credentials(orch: dict[str, Any]) -> list[Check]:
    """Live lanes must be able to RESOLVE a credential for every worker.

    A missing key does not raise at config time — the first worker call
    fails, silently producing zero provider traffic and unscoreable
    rollouts (r5 launch, 2026-07-28).
    """
    from ultra.providers import load_dotenv, resolve_api_key, routed_provider

    load_dotenv()
    out: list[Check] = []
    models: set[str] = set()
    for env in orch.get("env", []):
        args = env.get("args", {})
        if args.get("provider_mode") != "live":
            continue
        pilot = _read_json(_resolve(args.get("pilot_config_path", "")))
        if not pilot:
            continue
        for worker in (pilot.get("worker_pool") or {}).values():
            model = str(worker.get("model") or "")
            if model:
                models.add(model)
    missing: dict[str, str] = {}
    for model in sorted(models):
        try:
            key_env = str(routed_provider(model).get("key_env") or "")
        except Exception as exc:  # noqa: BLE001 — unroutable model is a failure
            missing[model] = f"unroutable: {exc}"
            continue
        if key_env and not resolve_api_key(key_env):
            missing[model] = f"{key_env} unresolved"
    out.append(Check(
        "workers.credentials_resolvable", not missing,
        f"{len(models)} pool models routable with credentials" if not missing
        else f"MISSING: {missing}"))
    return out


def check_absolute_paths(orch_path: Path) -> list[Check]:
    """Config paths must be ABSOLUTE.

    Env servers resolve relative paths against their own cwd, so a launch
    from any directory but the repo root dies with a missing pilot config
    — and preflight (which resolves against the repo root) would still
    pass. Requiring absolute paths removes the cwd dependency entirely.
    """
    relative = [
        line.strip() for line in orch_path.read_text().splitlines()
        if any(f"{k}:" in line for k in
               ("pilot_config_path", "task_manifest_path", "live_safety_path"))
        and not line.split(":", 1)[1].strip().startswith("/")
    ]
    return [Check("orch.paths_absolute", not relative,
                  f"all lane paths absolute" if not relative
                  else f"{len(relative)} relative path(s), e.g. {relative[0][:70]}")]


def check_lane_files(orch: dict[str, Any]) -> list[Check]:
    """Every lane's pilot config, task manifest and safety manifest must exist."""
    out: list[Check] = []
    for env in orch.get("env", []):
        args = env.get("args", {})
        name = env.get("name", "?")
        for key in ("pilot_config_path", "task_manifest_path", "live_safety_path"):
            value = args.get(key)
            if value is None:
                continue
            path = _resolve(value)
            ok = path.exists() and path.stat().st_size > 0
            out.append(Check(f"{name}.{key}", ok,
                             f"{path}" if ok else f"MISSING/EMPTY {path}"))
    return out


def check_live_safety(orch: dict[str, Any]) -> list[Check]:
    """Live lanes need an APPROVED manifest that covers the lane and its workers."""
    from ultra.live_worker_safety import VERSION as SAFETY_VERSION

    out: list[Check] = []
    for env in orch.get("env", []):
        args = env.get("args", {})
        name = env.get("name", "?")
        if args.get("provider_mode") != "live":
            continue
        safety_path = args.get("live_safety_path")
        pilot_path = args.get("pilot_config_path")
        if not safety_path or not pilot_path:
            out.append(Check(f"{name}.safety", False, "live lane without safety/pilot path"))
            continue
        safety = _read_json(_resolve(safety_path))
        pilot = _read_json(_resolve(pilot_path))
        if safety is None or pilot is None:
            missing = "safety manifest" if safety is None else "pilot config"
            out.append(Check(f"{name}.safety", False, f"unreadable {missing}"))
            continue
        lane = args.get("lane")
        problems = []
        if safety.get("version") != SAFETY_VERSION:
            problems.append(f"version {safety.get('version')!r} != {SAFETY_VERSION!r}")
        if not safety.get("approved"):
            problems.append("NOT APPROVED")
        if lane not in safety.get("allowed_lanes", []):
            problems.append(f"lane {lane!r} not allowed")
        workers = set(pilot.get("lane_worker_masks", {}).get(lane, []))
        allowed = set(safety.get("allowed_workers_by_lane", {}).get(lane, []))
        if not workers or not workers <= allowed:
            problems.append(f"workers {sorted(workers - allowed)} not allowed")
        if args.get("allow_yunwu_live"):
            problems.append("allow_yunwu_live is true")
        tasks = pilot.get("task_ids_by_lane", {}).get(lane, [])
        cap = safety.get("max_examples_by_lane", {}).get(lane)
        if cap is not None and len(tasks) > int(cap):
            problems.append(f"{len(tasks)} tasks > cap {cap}")
        out.append(Check(f"{name}.safety", not problems,
                         "; ".join(problems) if problems
                         else f"approved, {len(tasks)} tasks, workers {sorted(workers)}"))
    return out


def check_lane_tasks_present(orch: dict[str, Any]) -> list[Check]:
    """A lane's task ids must actually exist in the manifest it reads."""
    out: list[Check] = []
    for env in orch.get("env", []):
        args = env.get("args", {})
        name, lane = env.get("name", "?"), args.get("lane")
        manifest = args.get("task_manifest_path")
        pilot_path = args.get("pilot_config_path")
        if not manifest or not pilot_path:
            continue
        manifest_path = _resolve(manifest)
        pilot = _read_json(_resolve(pilot_path))
        if pilot is None:
            continue  # already reported by the safety/file checks
        wanted = set(pilot.get("task_ids_by_lane", {}).get(lane, []))
        if not wanted or not manifest_path.exists():
            continue
        have = {json.loads(line)["task_id"]
                for line in manifest_path.read_text().splitlines() if line.strip()}
        # The env's dataset is manifest ∩ authorized lane ids: several envs
        # may share one lane id list while each reads its own manifest slice,
        # and a manifest may deliberately carry more rows than the lane draws
        # (telecom ships all 2,171 train tasks; the config selects 240).
        # Unauthorized rows are never loaded, so the only fatal state is an
        # EMPTY intersection — that lane would contribute zero data.
        usable = have & wanted
        detail = f"{len(usable)} usable tasks (manifest {len(have)}, lane {len(wanted)})"
        if not usable:
            detail = "ZERO usable tasks — manifest and lane ids are disjoint"
        out.append(Check(f"{name}.tasks_in_manifest", bool(usable), detail))
    return out


def check_trainer_agreement(orch: dict[str, Any], train: dict[str, Any]) -> list[Check]:
    orch_len = orch.get("sequence_len")
    train_len = train.get("sequence_len") or train.get("data", {}).get("sequence_len")
    return [Check(
        "trainer.sequence_len_matches_orch",
        orch_len is not None and orch_len == train_len,
        f"orch={orch_len}, trainer={train_len}",
    )]


def check_serving(orch: dict[str, Any], timeout: float = 5.0) -> list[Check]:
    """The endpoint must answer to the EXACT model name the orch will send."""
    import urllib.request

    base = (orch.get("client", {}).get("base_url") or [""])[0]
    wanted = orch.get("model", {}).get("name")
    url = base.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            served = [m["id"] for m in json.loads(resp.read())["data"]]
    except Exception as exc:  # noqa: BLE001 — unreachable endpoint is a failed check
        return [Check("serving.model_name", False, f"{url}: {type(exc).__name__}")]
    return [Check("serving.model_name", wanted in served,
                  f"want {wanted!r}, served {served}")]


def run(orch_path: Path, train_path: Path, *, skip_serving: bool = False) -> dict[str, Any]:
    import yaml

    orch = yaml.safe_load(orch_path.read_text())
    train = yaml.safe_load(train_path.read_text())

    checks: list[Check] = []
    checks += check_orch_lanes(orch)
    checks += check_worker_credentials(orch)
    checks += check_absolute_paths(orch_path)
    checks += check_lane_files(orch)
    checks += check_live_safety(orch)
    checks += check_lane_tasks_present(orch)
    checks += check_trainer_agreement(orch, train)
    if not skip_serving:
        checks += check_serving(orch)

    failed = [c for c in checks if not c.ok]
    return {
        "orch": str(orch_path),
        "trainer": str(train_path),
        "checks": [c.as_dict() for c in checks],
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "ok": not failed,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orch", type=Path, default=MANIFEST_DIR / "orch_r5.yaml")
    ap.add_argument("--trainer", type=Path,
                    default=ROOT / "director/manifests/fugu_clean_v1/grpo_pilot_train/train_r5.yaml")
    ap.add_argument("--skip-serving", action="store_true",
                    help="skip the endpoint check (sidecar not started yet)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    report = run(args.orch, args.trainer, skip_serving=args.skip_serving)
    if args.json:
        print(json.dumps(report, indent=1))
    else:
        for check in report["checks"]:
            print(f"[{'PASS' if check['ok'] else 'FAIL'}] {check['check']}: {check['detail']}")
        print(f"\n{report['passed']} passed, {report['failed']} failed — "
              f"{'READY TO LAUNCH' if report['ok'] else 'DO NOT LAUNCH'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
