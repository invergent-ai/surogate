"""Scoped Docker cleanup helpers for Harbor/TaskTrove runs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DOCKER_NETWORK_JANITOR_VERSION = "fugu_ultra_docker_network_janitor_v1"

DEFAULT_COMPOSE_PROJECT_PREFIXES = (
    "code_contests-",
    "inferredbugs-",
    "llm-verifier-freelancer-",
    "multifile-",
    "nl2bash-",
    "pymethods2test-",
    "r2egym-",
    "r2egym-v",
    "stack-bash-",
    "swe-smith-",
    "swegym-",
    "swesmith-",
    "tasktrove-",
)

PROTECTED_NETWORK_NAMES = {"bridge", "host", "none"}


def _run_docker(args: list[str], *, docker_bin: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [docker_bin, *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _docker_available(docker_bin: str) -> bool:
    return bool(shutil.which(docker_bin) or Path(docker_bin).exists())


def _load_json_lines(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            rows.append(data)
    return rows


def _matching_project(project: str | None, name: str, prefixes: tuple[str, ...], *, all_compose: bool) -> bool:
    if all_compose:
        return True
    values = [name]
    if project:
        values.append(project)
    return any(value.startswith(prefix) for value in values for prefix in prefixes)


def stale_compose_network_candidate(
    network: dict[str, Any],
    *,
    prefixes: tuple[str, ...] = DEFAULT_COMPOSE_PROJECT_PREFIXES,
    all_compose: bool = False,
) -> tuple[bool, str]:
    """Return whether a Docker Compose network is a safe stale-cleanup candidate."""

    name = str(network.get("Name") or network.get("Name".lower()) or "")
    labels = network.get("Labels")
    if not isinstance(labels, dict):
        labels = {}
    project = labels.get("com.docker.compose.project")
    if name in PROTECTED_NETWORK_NAMES:
        return False, "protected"
    if not project:
        return False, "not_compose"
    if not _matching_project(str(project), name, prefixes, all_compose=all_compose):
        return False, "unmatched"
    containers = network.get("Containers")
    if isinstance(containers, dict) and containers:
        return False, "active"
    return True, "stale"


def _network_summary(network: dict[str, Any]) -> dict[str, Any]:
    labels = network.get("Labels")
    if not isinstance(labels, dict):
        labels = {}
    containers = network.get("Containers")
    return {
        "id": str(network.get("Id") or "")[:12],
        "name": str(network.get("Name") or ""),
        "project": labels.get("com.docker.compose.project"),
        "container_count": len(containers) if isinstance(containers, dict) else 0,
    }


def _list_compose_networks(*, docker_bin: str) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    proc = _run_docker(
        ["network", "ls", "--filter", "label=com.docker.compose.project", "--format", "{{json .}}"],
        docker_bin=docker_bin,
    )
    if proc.returncode != 0:
        return [], [proc.stderr.strip() or proc.stdout.strip() or "docker network ls failed"]
    rows = _load_json_lines(proc.stdout)
    networks: list[dict[str, Any]] = []
    for row in rows:
        network_id = str(row.get("ID") or row.get("Id") or "")
        if not network_id:
            continue
        inspect = _run_docker(["network", "inspect", network_id], docker_bin=docker_bin)
        if inspect.returncode != 0:
            errors.append(inspect.stderr.strip() or f"docker network inspect failed for {network_id}")
            continue
        try:
            data = json.loads(inspect.stdout)
        except json.JSONDecodeError:
            errors.append(f"docker network inspect returned invalid JSON for {network_id}")
            continue
        if isinstance(data, list) and data and isinstance(data[0], dict):
            networks.append(data[0])
    return networks, errors


def cleanup_stale_docker_networks(
    *,
    dry_run: bool = True,
    max_remove: int = 200,
    prefixes: tuple[str, ...] = DEFAULT_COMPOSE_PROJECT_PREFIXES,
    all_compose: bool = False,
    docker_bin: str = "docker",
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Remove detached Harbor/TaskTrove Docker Compose networks.

    This intentionally does not run ``docker network prune``. It only considers
    Compose networks, applies a Harbor/TaskTrove name allowlist by default, and
    skips every network with attached containers.
    """

    created_at_utc = created_at_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    report: dict[str, Any] = {
        "version": DOCKER_NETWORK_JANITOR_VERSION,
        "created_at_utc": created_at_utc,
        "mode": "dry_run" if dry_run else "delete",
        "docker_bin": docker_bin,
        "filters": {
            "prefixes": list(prefixes),
            "all_compose": all_compose,
            "max_remove": max_remove,
        },
        "docker_available": _docker_available(docker_bin),
        "networks_seen": 0,
        "stale_candidates": 0,
        "removed": 0,
        "failed_removals": 0,
        "skipped": {},
        "stale_networks": [],
        "removed_networks": [],
        "errors": [],
    }
    if not report["docker_available"]:
        report["errors"].append(f"{docker_bin} not found")
        return report

    networks, errors = _list_compose_networks(docker_bin=docker_bin)
    report["errors"].extend(errors)
    report["networks_seen"] = len(networks)
    skipped: Counter[str] = Counter()
    stale: list[dict[str, Any]] = []
    for network in networks:
        is_candidate, reason = stale_compose_network_candidate(
            network,
            prefixes=prefixes,
            all_compose=all_compose,
        )
        if is_candidate:
            stale.append(network)
        else:
            skipped[reason] += 1

    report["skipped"] = dict(sorted(skipped.items()))
    report["stale_candidates"] = len(stale)
    report["stale_networks"] = [_network_summary(network) for network in stale[:max_remove]]
    if dry_run:
        return report

    for network in stale[:max_remove]:
        network_id = str(network.get("Id") or network.get("ID") or network.get("Name") or "")
        if not network_id:
            report["failed_removals"] += 1
            report["errors"].append(f"network had no removable id: {_network_summary(network)}")
            continue
        proc = _run_docker(["network", "rm", network_id], docker_bin=docker_bin)
        if proc.returncode == 0:
            report["removed"] += 1
            report["removed_networks"].append(_network_summary(network))
        else:
            report["failed_removals"] += 1
            report["errors"].append(proc.stderr.strip() or f"docker network rm failed for {network_id}")
    return report


def run_cli(args: argparse.Namespace) -> dict[str, Any]:
    prefixes = tuple(args.name_prefix) if args.name_prefix else DEFAULT_COMPOSE_PROJECT_PREFIXES
    report = cleanup_stale_docker_networks(
        dry_run=not args.delete,
        max_remove=args.max_remove,
        prefixes=prefixes,
        all_compose=args.all_compose,
        docker_bin=args.docker_bin,
    )
    if args.report_out:
        out = Path(args.report_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
