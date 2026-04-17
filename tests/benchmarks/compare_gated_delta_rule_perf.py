"""Compare surogate and FLA gated-delta-rule benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def _key(c: dict) -> tuple[int, int, int, int, int]:
    return int(c["B"]), int(c["T"]), int(c["H"]), int(c["K"]), int(c["V"])


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    art = repo_root / "tests" / "artifacts"
    suro_path = art / "gated_delta_rule_surogate_perf.json"
    fla_path = art / "gated_delta_rule_fla_perf.json"

    suro = json.loads(suro_path.read_text())
    fla = json.loads(fla_path.read_text())

    su_cases = {_key(c): c for c in suro["cases"]}
    fl_cases = {_key(c): c for c in fla["cases"]}

    common = sorted(set(su_cases).intersection(fl_cases))
    if not common:
        raise RuntimeError("No common benchmark cases between surogate and FLA artifacts.")

    required_cases = {
        (1, 512, 8, 128, 128),  # dev
        (2, 2048, 16, 128, 128),  # production-shaped
    }
    missing = sorted(required_cases.difference(common))
    if missing:
        raise RuntimeError(f"Missing required 128x128 benchmark cases: {missing}")

    print("B T H K V     | su_fwd  fla_fwd  speedup | su_bwd  fla_bwd  speedup | su_total fla_total speedup")
    perf_failures: list[tuple[tuple[int, int, int, int, int], float, float]] = []
    for k in common:
        su = su_cases[k]
        fl = fl_cases[k]
        sf, ff = float(su["forward_ms"]), float(fl["forward_ms"])
        sb, fb = float(su["backward_ms"]), float(fl["backward_ms"])
        st, ft = float(su["total_ms"]), float(fl["total_ms"])
        print(
            f"{k[0]} {k[1]} {k[2]} {k[3]} {k[4]} | "
            f"{sf:7.3f} {ff:7.3f} {ff / sf:7.2f}x | "
            f"{sb:7.3f} {fb:7.3f} {fb / sb:7.2f}x | "
            f"{st:8.3f} {ft:8.3f} {ft / st:7.2f}x"
        )
        if k in required_cases and st > ft:
            perf_failures.append((k, st, ft))

    if perf_failures:
        msg = ", ".join(f"{k}: su_total={st:.3f}ms > fla_total={ft:.3f}ms" for k, st, ft in perf_failures)
        raise RuntimeError(f"Performance gate failed for required cases: {msg}")


if __name__ == "__main__":
    main()
