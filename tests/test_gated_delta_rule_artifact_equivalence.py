"""Compare deterministic surogate vs FLA gated-delta-rule value artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

FIELDS = (
    "out",
    "final_state",
    "d_query",
    "d_key",
    "d_value",
    "d_g",
    "d_beta",
    "d_initial_state",
)


def _case_key(case: dict) -> tuple[str, int, int, int, int, int]:
    meta = case["meta"]
    return (
        str(meta.get("case_name", "")),
        int(meta["B"]),
        int(meta["T"]),
        int(meta["H"]),
        int(meta["K"]),
        int(meta["V"]),
    )


def _case_map(report: dict) -> dict[tuple[str, int, int, int, int, int], dict]:
    if "cases" in report:
        return {_case_key(case): case for case in report["cases"]}
    return {_case_key(report): report}


def _field_tolerance(name: str) -> tuple[float, float]:
    if name in {"final_state", "d_initial_state"}:
        return 1e-2, 1e-2
    return 3e-2, 3e-2


def _assert_values_close(name: str, suro: dict, fla: dict) -> None:
    sv = suro.get("values")
    fv = fla.get("values")
    assert sv is not None and fv is not None, f"{name}: missing serialized values in artifacts"

    su = np.asarray(sv, dtype=np.float32)
    fl = np.asarray(fv, dtype=np.float32)
    assert su.shape == fl.shape, f"{name}: shape mismatch {su.shape} vs {fl.shape}"

    atol, rtol = _field_tolerance(name)
    if not np.allclose(su, fl, atol=atol, rtol=rtol):
        abs_diff = np.abs(su - fl)
        rel_diff = abs_diff / np.maximum(np.abs(fl), 1e-6)
        idx = int(abs_diff.argmax())
        raise AssertionError(
            f"{name}: mismatch (atol={atol}, rtol={rtol}); "
            f"max_abs={abs_diff[idx]:.6f} max_rel={rel_diff.max():.6f} at flat_idx={idx}, "
            f"suro={su[idx]:.6f}, fla={fl[idx]:.6f}"
        )


def _assert_stats_consistent(name: str, suro: dict, fla: dict) -> None:
    for key in ("numel",):
        assert int(suro[key]) == int(fla[key]), f"{name}.{key}: mismatch {suro[key]} vs {fla[key]}"


def test_gated_delta_rule_surogate_matches_fla_artifacts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    art_dir = repo_root / "tests" / "artifacts"
    suro_path = art_dir / "gated_delta_rule_surogate_values.json"
    fla_path = art_dir / "gated_delta_rule_fla_values.json"

    if not suro_path.exists() or not fla_path.exists():
        pytest.skip(
            "Missing artifacts. Generate both first:\n"
            "  1) pytest -q tests/test_gated_delta_rule_reference.py\n"
            "  2) ./csrc/build/unit-tests \"[kernels][gated_delta_rule][dump]\""
        )

    suro_report = json.loads(suro_path.read_text())
    fla_report = json.loads(fla_path.read_text())

    su_cases = _case_map(suro_report)
    fl_cases = _case_map(fla_report)
    assert su_cases.keys() == fl_cases.keys(), "Case sets differ between surogate and FLA artifacts"

    for case_key in sorted(su_cases):
        su_case = su_cases[case_key]
        fl_case = fl_cases[case_key]
        for field in FIELDS:
            _assert_stats_consistent(field, su_case[field], fl_case[field])
            _assert_values_close(field, su_case[field], fl_case[field])
