"""Lane T: the verdict of the capped session, from its two reports, by the PASS line written in lanes/T/NOTES.md BEFORE the session:
(1) the 28-32k row completes its update without running out of memory; (2) the step's tok/s is measured at ~16k and ~32k, with 8k beside
them; (3) the capped arm meets the unchanged line (cos >= 0.9999, rel err <= 1 %) at 1k / 4k / 8k vs A1, A0 and ift_backward, with A/A
exactly 0.0, finite. Anything else is FAIL.
Usage: python cap_verdict.py kit_rows.json lever_check.json [arm, default F4C]   (exit 0 = PASS, 1 = FAIL)
"""

from __future__ import annotations

import json
import sys

COS, REL = 0.9999, 0.01


def verdict(kit, lev, arm="F4C"):
    why = []
    rows = kit.get("rows", [])
    if kit.get("oom"):
        why.append(f"out of memory on the row of T = {kit['oom'].get('row_T')}")
    if not kit.get("all_rows_done"):
        why.append("not every kit row completed")
    if not kit.get("finite"):
        why.append("non-finite gradient or no gradient")
    if not any(28000 <= r["T"] <= 32768 for r in rows):
        why.append("no completed row of 28-32k tokens")
    near16 = [r for r in rows if 14000 <= r["T"] <= 18000]
    near32 = [r for r in rows if 28000 <= r["T"] <= 32768]
    if not near16:
        why.append("no measured row near 16k")
    if not near32:
        why.append("no measured row near 32k")
    by = lev.get("by_length", {})
    for T in ("1024", "4096", "8192"):
        ar = by.get(T, {}).get("arms", {}).get(arm)
        if ar is None:
            why.append(f"{arm} not measured at {T}")
            continue
        if not ar.get("finite"):
            why.append(f"{arm} not finite at {T}")
        aa = ar.get("aa_floor") or {}
        if aa.get("rel_err_max") != 0.0:
            why.append(f"{arm} A/A at {T} = {aa.get('rel_err_max')} (must be exactly 0.0)")
        keys = ["vs_previous", "vs_A0"] + (["vs_ift_backward"] if T == "1024" else [])
        for k in keys:
            c = ar.get(k)
            if not c:
                why.append(f"{arm} has no {k} at {T}")
            elif c["cos_min"] < COS or c["rel_err_max"] > REL:
                why.append(f"{arm} {k} at {T}: cos {c['cos_min']:.6f}, rel err {c['rel_err_max'] * 100:.3f} %")
    speed = {"8k_tok_per_s": (by.get("8192", {}).get("arms", {}).get(arm) or {}).get("tok_per_s"),
             "16k_row": near16[0] if near16 else None, "32k_row": near32[-1] if near32 else None}
    return not why, why, speed


def main():
    kit = json.load(open(sys.argv[1]))
    lev = json.load(open(sys.argv[2]))
    arm = sys.argv[3] if len(sys.argv) > 3 else "F4C"
    ok, why, speed = verdict(kit, lev, arm)
    print(json.dumps({"speed": speed, "long_row_check": lev.get("long")}, default=float))
    print("CAP PASS" if ok else "CAP FAIL: " + "; ".join(why), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
