#!/usr/bin/env python3
"""Decision rows against a served /v1/decisions endpoint (standard library only).
  decisions_eval.py ROWS.jsonl --urls http://h:p[,...] --model NAME --out DIR [--mode balanced] [--limit N] [--concurrency 8] [--manifest manifest.json --split decision-validation.jsonl]
ROWS: {"request": {"state", "questions"}, "target": {"answers": {q: {"type", "choice" | "noul" | "score"}}}}. The request is sent as given (+ model, mode when set).
Scored: a choice answer is right when its "choice" equals the target's; a noul answer when (noul >= 0.5) equals (target == "1"). Per question name, per source family
(manifest row ranges), the mean confidence of the answer given and ECE (10 bins); system15.loops_used and escalate_prob when the reply carries them."""
import argparse, concurrent.futures as cf, http.client, json, os, sys, time, urllib.parse


def post(url, body, timeout=600):
    u = urllib.parse.urlparse(url)
    for attempt in range(3):
        try:
            c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
            c.request("POST", "/v1/decisions", json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            data = r.read()
            c.close()
            return r.status, data
        except (OSError, http.client.HTTPException) as e:
            err = repr(e)
            time.sleep(1 + attempt)
    return -1, err.encode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows"); ap.add_argument("--urls", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default=""); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--manifest", default=""); ap.add_argument("--split", default="")
    ap.add_argument("--families", default="", help="send only the rows of these source families (comma list; needs --manifest)"); ap.add_argument("--max-loops", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(x) for x in open(a.rows)]
    fam = [None] * len(rows)
    if a.manifest:
        i = 0
        for s in json.load(open(a.manifest))[a.split]["sources_in_row_order"]:
            for _ in range(s["rows"]):
                if i < len(rows):
                    fam[i] = s["file"].split("/")[0]
                i += 1
        assert i == len(rows), (i, len(rows))
    if a.limit:
        step = max(1, len(rows) // a.limit)                    # an even spread over the file (the rows are grouped by source)
        idx = list(range(0, len(rows), step))[:a.limit]
    else:
        idx = list(range(len(rows)))
    if a.families:
        keep = set(a.families.split(","))
        idx = [i for i in idx if fam[i] in keep]
    urls = a.urls.split(",")
    os.makedirs(a.out, exist_ok=True)

    def one(k):
        i = idx[k]
        body = dict(rows[i]["request"])
        body["model"] = a.model
        if a.mode:
            body["mode"] = a.mode
        if a.max_loops:
            body["max_loops"] = a.max_loops
        t = time.time()
        st, data = post(urls[k % len(urls)], body)
        return i, st, data, time.time() - t

    t0 = time.time()
    res = {}
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        for i, st, data, dt in ex.map(one, range(len(idx))):
            res[i] = (st, data, dt)
    wall = time.time() - t0
    per, bad = [], 0
    with open(os.path.join(a.out, "replies.jsonl"), "w") as f:
        for i in idx:
            st, data, dt = res[i]
            if st != 200:
                bad += 1
                f.write(json.dumps({"row": i, "status": st, "error": data[:300].decode("utf-8", "replace")}) + "\n")
                continue
            rep = json.loads(data)
            f.write(json.dumps({"row": i, "family": fam[i], "seconds": dt, "answers": rep.get("answers"), "usage": rep.get("usage")}) + "\n")
            for q, tgt in rows[i]["target"]["answers"].items():
                ans = (rep.get("answers") or {}).get(q) or {}
                if tgt["type"] == "choice":
                    ok = ans.get("choice") == tgt["choice"]
                    pr = ans.get("probabilities") or {}
                    conf = max(pr.values()) if pr else None
                elif tgt["type"] == "noul":
                    p = ans.get("noul")
                    ok = p is not None and ((p >= 0.5) == (str(tgt["noul"]) == "1"))
                    conf = None if p is None else max(p, 1 - p)
                else:
                    continue
                s15 = ans.get("system15") or {}
                per.append({"row": i, "family": fam[i], "q": q, "type": tgt["type"], "ok": bool(ok), "conf": conf, "loops": s15.get("loops_used"), "esc": s15.get("escalate_prob")})

    def summ(sel):
        n = len(sel)
        if not n:
            return None
        acc = sum(x["ok"] for x in sel) / n
        cs = [x for x in sel if x["conf"] is not None]
        ece = 0.0
        for b in range(10):
            bucket = [x for x in cs if (b / 10 <= x["conf"] < (b + 1) / 10) or (b == 9 and x["conf"] == 1.0)]
            if bucket:
                ece += len(bucket) / len(cs) * abs(sum(x["ok"] for x in bucket) / len(bucket) - sum(x["conf"] for x in bucket) / len(bucket))
        lo = [x["loops"] for x in sel if x["loops"] is not None]
        return {"n": n, "accuracy": acc, "ece": ece if cs else None, "mean_conf": sum(x["conf"] for x in cs) / len(cs) if cs else None, "mean_loops": sum(lo) / len(lo) if lo else None}

    out = {"rows_sent": len(idx), "failed": bad, "wall_s": wall, "model": a.model, "mode": a.mode, "by_question": {}, "by_family": {}}
    for q in sorted({x["q"] for x in per}):
        out["by_question"][q] = summ([x for x in per if x["q"] == q])
    both = {}
    for x in per:
        both.setdefault(x["row"], []).append(x["ok"])
    out["rows_all_questions_right"] = sum(all(v) for v in both.values()) / max(len(both), 1)
    for fm in sorted({x["family"] for x in per if x["family"]}):
        out["by_family"][fm] = {q: (summ([x for x in per if x["family"] == fm and x["q"] == q]) or {}).get("accuracy") for q in out["by_question"]}
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "by_family"}))
    return 0 if bad == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
