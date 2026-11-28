#!/usr/bin/env python3
"""Paired comparison of two decision adapters on the same rows, from the replies of decisions_eval.py (same rows file, same serving options).
  decisions_compare.py ROWS.jsonl MANIFEST.json SPLIT A_replies.jsonl B_replies.jsonl OUT_DIR [--names OLD NEW] [--slice-family NAME --slice-kinds a,b]
Writes OUT_DIR/paired-predictions.jsonl (one line a row: family, targets, both adapters' answers, probabilities, loops; no request text) and
OUT_DIR/summary.json (overall and by family: action / attack / both, paired differences with 95 % bootstrap intervals over rows; families that got
worse, with example rows; with --slice-family, the rows of that family whose state.observations[].kind set equals --slice-kinds: false positives
of its benign rows (target action "watch", attack 0) and misses of its hostile rows). The two questions are named "action" (choice) and "attack"
(yes / no), with "watch" as the no-action choice."""
import argparse
import collections
import json
import os

import numpy as np


def load_replies(p):
    out = {}
    for line in open(p):
        r = json.loads(line)
        if r.get("answers"):
            out[r["row"]] = r["answers"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows"); ap.add_argument("manifest"); ap.add_argument("split"); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("out")
    ap.add_argument("--names", nargs=2, default=["A", "B"]); ap.add_argument("--seed", type=int, default=20261127)
    ap.add_argument("--slice-family", default=""); ap.add_argument("--slice-kinds", default="")
    a = ap.parse_args()
    na, nb = a.names
    rows = [json.loads(x) for x in open(a.rows)]
    fam = []
    for s in json.load(open(a.manifest))[a.split]["sources_in_row_order"]:
        fam += [s["file"].split("/")[0]] * s["rows"]
    A, B = load_replies(a.a), load_replies(a.b)
    ids = sorted(set(A) & set(B))
    os.makedirs(a.out, exist_ok=True)
    rec = []
    for i in ids:
        t = rows[i]["target"]["answers"]
        ta, tk = t["action"]["choice"], str(t["attack"]["noul"]) == "1"
        kinds = sorted({o.get("kind") for o in rows[i]["request"]["state"].get("observations", []) if isinstance(o, dict)})
        d = {"row": i, "family": fam[i], "observation_kinds": kinds, "target_action": ta, "target_attack": int(tk)}
        for n, R in ((na, A), (nb, B)):
            x = R[i]
            d[n] = {"action": x["action"].get("choice"), "action_confidence": x["action"].get("confidence"), "attack_probability": x["attack"].get("noul"),
                    "attack": int(x["attack"]["noul"] >= 0.5), "loops_action": (x["action"].get("system15") or {}).get("loops_used"),
                    "loops_attack": (x["attack"].get("system15") or {}).get("loops_used")}
            d[n]["action_ok"] = int(d[n]["action"] == ta)
            d[n]["attack_ok"] = int(bool(d[n]["attack"]) == tk)
            d[n]["both_ok"] = int(d[n]["action_ok"] and d[n]["attack_ok"])
        rec.append(d)
    with open(os.path.join(a.out, "paired-predictions.jsonl"), "w") as f:
        for d in rec:
            f.write(json.dumps(d) + "\n")
    rng = np.random.default_rng(a.seed)

    def stats(sub):
        o = {"n": len(sub)}
        for k in ("action_ok", "attack_ok", "both_ok"):
            x = np.array([d[na][k] for d in sub], float); y = np.array([d[nb][k] for d in sub], float)
            dd = y - x
            bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)] if len(dd) else [0.0]
            o[k.replace("_ok", "")] = {na: float(x.mean()), nb: float(y.mean()), "diff": float(dd.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                         "fixed": int(((x == 0) & (y == 1)).sum()), "broken": int(((x == 1) & (y == 0)).sum())}
        return o

    out = {"rows": len(rec), "names": [na, nb], "overall": stats(rec), "by_family": {}}
    for fm in sorted({d["family"] for d in rec}):
        out["by_family"][fm] = stats([d for d in rec if d["family"] == fm])
    kinds = sorted(k for k in a.slice_kinds.split(",") if k)
    sl = [d for d in rec if a.slice_family and d["family"] == a.slice_family and (not kinds or d["observation_kinds"] == kinds)]
    ben = [d for d in sl if d["target_action"] == "watch" and d["target_attack"] == 0]
    hos = [d for d in sl if d["target_action"] != "watch" and d["target_attack"] == 1]
    out["slice"] = {"family": a.slice_family, "kinds": kinds, "benign_rows": len(ben), "hostile_rows": len(hos)}
    for n in (na, nb):
        out["slice"][n] = {
            "false_positive_action": sum(d[n]["action"] != "watch" for d in ben), "false_positive_attack": sum(d[n]["attack"] == 1 for d in ben),
            "false_positive_either": sum(d[n]["action"] != "watch" or d[n]["attack"] == 1 for d in ben),
            "miss_action": sum(d[n]["action"] == "watch" for d in hos), "miss_attack": sum(d[n]["attack"] == 0 for d in hos),
            "miss_either": sum(d[n]["action"] == "watch" or d[n]["attack"] == 0 for d in hos),
            "mean_attack_probability_benign": float(np.mean([d[n]["attack_probability"] for d in ben])) if ben else None,
            "mean_attack_probability_hostile": float(np.mean([d[n]["attack_probability"] for d in hos])) if hos else None}
    reg = []
    for fm, s in out["by_family"].items():
        worse = [k for k in ("action", "attack", "both") if s[k]["diff"] < 0 and s[k]["ci95"][1] < 0]
        lower = [k for k in ("action", "attack", "both") if s[k]["diff"] < 0]
        if lower:
            ex = [d for d in rec if d["family"] == fm and (d[na]["both_ok"] and not d[nb]["both_ok"])][:4]
            reg.append({"family": fm, "lower_on": lower, "interval_below_zero_on": worse, "n": s["n"],
                        **{k: {na: s[k][na], nb: s[k][nb], "diff": s[k]["diff"], "ci95": s[k]["ci95"]} for k in ("action", "attack", "both")},
                        "example_rows_right_before_wrong_after": [{"row": d["row"], "target_action": d["target_action"], "target_attack": d["target_attack"],
                                                                   na: {k: d[na][k] for k in ("action", "attack", "attack_probability")},
                                                                   nb: {k: d[nb][k] for k in ("action", "attack", "attack_probability")}} for d in ex]})
    out["families_lower"] = reg
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    o = out["overall"]
    print(f"rows {len(rec)}; " + "; ".join(f"{k}: {o[k][na] * 100:.1f} -> {o[k][nb] * 100:.1f} ({o[k]['diff'] * 100:+.1f} pp [{o[k]['ci95'][0] * 100:+.1f}, {o[k]['ci95'][1] * 100:+.1f}], fixed {o[k]['fixed']}, broken {o[k]['broken']})" for k in ("action", "attack", "both")))
    if a.slice_family:
        print("slice:", json.dumps(out["slice"]))
    print(f"{'family':30s}   n   action {na}->{nb}    attack {na}->{nb}    both {na}->{nb}")
    for fm, s in sorted(out["by_family"].items(), key=lambda kv: kv[1]["both"]["diff"]):
        print(f"{fm:30s} {s['n']:4d}   " + "   ".join(f"{s[k][na] * 100:5.1f} -> {s[k][nb] * 100:5.1f} ({s[k]['diff'] * 100:+5.1f})" for k in ("action", "attack", "both")))
    print("families lower on any measure:", [(r["family"], r["lower_on"], "CI<0: " + ",".join(r["interval_below_zero_on"])) for r in reg])


if __name__ == "__main__":
    main()
