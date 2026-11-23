"""The 8-card gate's row file and its round-trip verdict on synthetic log-probs (no GPU)."""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t8_gate8 as G  # noqa: E402


class A:
    pass


def verdict(lp_t, lp_e, base):
    d = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    np.savez(os.path.join(d, "trainer_v1.npz"), **{f"lp_{i}": x for i, x in enumerate(lp_t)})
    json.dump([{"lp": [0.0] + [float(v) for v in x]} for x in lp_e], open(os.path.join(d, "engine_v1.json"), "w"))
    out = os.path.join(d, "gate8.json")
    json.dump(base, open(out, "w"))
    a = A(); a.run_dir, a.out = d, out
    rc = G.phase_compare(a)
    return rc, json.load(open(out))


def main():
    rows, groups = G.load(sys.argv[1])
    assert len(groups["batch"]) == 32 and len(groups["speed"]) == 8 and len(groups["aa"]) == 8 and len(groups["rt"]) == 3
    assert all(r.T <= 58010 and r.loss_mask.any() and not r.loss_mask[0] and r.notes["weight"] == 1.0 and r.routed is None for r in rows)
    assert max(rows[i].T for i in groups["batch"]) > 57000
    g = np.random.default_rng(0)
    lp = [g.normal(-1, 1, 50).astype(np.float32) for _ in range(3)]
    ok = {"G8a": True, "G8b": True, "G8c": True, "G8d": True, "update_changed_logprobs": 5}
    rc, r = verdict(lp, lp, ok)
    assert rc == 0 and r["pass"] and r["G9"]["differ"] == 0 and r["G9"]["equal"] == 150
    bad = [x.copy() for x in lp]
    bad[1][7] = np.nextafter(bad[1][7], np.float32(0))
    rc, r = verdict(lp, bad, ok)
    assert rc != 0 and not r["G9"]["pass"] and r["G9"]["differ"] == 1, "one ulp must fail"
    rc, r = verdict(lp, lp, {**ok, "update_changed_logprobs": 0})
    assert rc != 0 and not r["G9"]["pass"], "an update that changed nothing is a vacuous round trip"
    rc, r = verdict(lp, lp, {**ok, "G8b": False})
    assert rc != 0 and not r["pass"]
    print("T8 GATE8 CPU TEST: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
