# usage: shared_prefix_check.py fork|ref <base_url> <out.json> [prompts.json]
#        shared_prefix_check.py compare <fork.json> <ref.json>
#        shared_prefix_check.py agents <base_url> <out.json>
# Shared prompt prefixes under MTP (csrc/src/serve/PATCHES.md #150), end to end against a running engine (model id "bench").
#  fork:    five prompts share exactly their first T tokens (T a multiple of the 2,048-token chunk, so chunking is the
#           same with and without the cache) and differ at token T, so a fork's MTP bridge pairs the kept hidden with
#           a different next token than the capturing prompt's. A1 is seen, A2 captures at T, A2 again, A3, A4, A5
#           fork. Greedy, 160 tokens, first-token top-5 logprobs. Writes the prompts and outputs.
#  ref:     the same prompts on an engine with SUROGATE_SERVE_SHARED_PREFIX_SLOTS=0 (no fork).
#  compare: fork against ref, prompt by prompt: first-token top-5 logprobs and the generated token ids.
#  agents:  4 concurrent agents x 2 waves, each a new conversation over the same ~12k-token preamble and 3 turns
#           (streamed, TTFT per turn); errors and TTFTs.
import json, sys, threading, time, urllib.request

def post(url, path, body, timeout=1800):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

WORDS = ("ledger invoice shipment warehouse customer region quarter forecast budget contract vendor audit payroll "
         "inventory schedule ticket incident release deployment cluster metric alert dashboard report summary").split()

def preamble():
    lines = ["You are an operations agent for a mid-sized logistics company. Follow the instructions exactly, call tools "
             "when a task needs data, never invent values, and keep answers short and factual.", "", "# Tools", ""]
    for i in range(64):
        w = [WORDS[(i * 7 + k) % len(WORDS)] for k in range(6)]
        lines.append(f"## tool_{i:02d}_{w[0]}_{w[1]}")
        lines.append(f"Looks up the {w[0]} {w[1]} records for a {w[2]} and returns the matching {w[3]} entries with their "
                     f"{w[4]} and {w[5]} fields, newest first. Use it when the user asks about a {w[2]} {w[0]} or needs "
                     f"the {w[5]} of a {w[1]}; do not use it for {w[3]} totals across regions.")
        lines.append("Parameters: " + json.dumps({"type": "object", "properties": {
            f"{w[2]}_id": {"type": "string", "description": f"Identifier of the {w[2]}"},
            "from_date": {"type": "string", "description": "ISO date, inclusive"},
            "to_date": {"type": "string", "description": "ISO date, inclusive"},
            "limit": {"type": "integer", "description": f"Most {w[3]} entries to return, at most 200"}},
            "required": [f"{w[2]}_id"]}))
        lines.append("")
    return "\n".join(lines)

USERS = ["Alpha team asks: which tool lists the overdue invoices of a customer, and what does it need?",
         "Bravo desk wants the shipment records of warehouse W-17 for last March; plan the tool calls.",
         "Charlie: explain in two sentences how you would find the budget forecast for region North.",
         "Delta needs the payroll audit entries of vendor V-202 since January; which tool and arguments?",
         "Echo reports a deployment incident on cluster C3; which tools help to investigate it, and why?"]

def tokenize(url, user):
    msgs = [{"role": "system", "content": preamble()}, {"role": "user", "content": user}]
    return post(url, "/tokenize", {"model": "bench", "messages": msgs,
                                   "chat_template_kwargs": {"enable_thinking": False}})["tokens"]

def build_prompts(url):
    ids = [tokenize(url, u) for u in USERS]
    common = min(len(x) for x in ids)
    for i in range(min(len(x) for x in ids)):
        if len({x[i] for x in ids}) > 1: common = i; break
    T = common // 2048 * 2048
    tails = [x[common:] for x in ids]
    if len({t[0] for t in tails}) != len(tails): raise SystemExit("user tails do not differ at their first token")
    return T, [ids[0][:T] + t for t in tails]

def generate(url, tokens):
    t0 = time.time()
    r = post(url, "/v1/chat/completions", {"model": "bench", "messages": [{"role": "user", "content": "x"}],
             "tokens": tokens, "temperature": 0, "max_tokens": 160, "logprobs": True, "top_logprobs": 5,
             "return_token_ids": True})
    c = r["choices"][0]
    lp = (c.get("logprobs") or {}).get("content") or []
    first = [(t["token"], t["logprob"]) for t in lp[0]["top_logprobs"]] if lp else []
    gen = c.get("token_ids") or r.get("token_ids") or [t["token"] for t in lp]
    return {"seconds": round(time.time() - t0, 2), "text": c["message"].get("content") or "", "first_top5": first,
            "generated": gen, "usage": r.get("usage")}

def run_prompts(url, out, prompts_file):
    if prompts_file:
        P = json.load(open(prompts_file)); T, prompts = P["T"], P["prompts"]
    else:
        T, prompts = build_prompts(url)
    order = [("A1", 0), ("A2", 1), ("A2again", 1), ("A3", 2), ("A4", 3), ("A5", 4)]
    res = {"T": T, "prompts": prompts, "runs": {}}
    print("T %d, prompt lengths %s" % (T, [len(p) for p in prompts]))
    for name, i in order:
        res["runs"][name] = generate(url, prompts[i])
        g = res["runs"][name]
        print("%-8s %6.2f s  %3d tokens  %r" % (name, g["seconds"], len(g["generated"]), g["text"][:90]))
    json.dump(res, open(out, "w"))

def compare(a_file, b_file):
    A, B = json.load(open(a_file))["runs"], json.load(open(b_file))["runs"]
    bad = 0
    pairs = [("A2", "A2again", A, A)] + [(n, n, A, B) for n in ("A2", "A3", "A4", "A5")]
    for x, y, X, Y in pairs:
        gx, gy = X[x]["generated"], Y[y]["generated"]
        div = next((k for k in range(min(len(gx), len(gy))) if gx[k] != gy[k]), None)
        same_first = X[x]["first_top5"] == Y[y]["first_top5"]
        label = f"{x}{'(fork)' if X is A else ''} vs {y}{'(ref)' if Y is B else '(fork)'}"
        ok = same_first and div is None and len(gx) == len(gy)
        bad += 0 if ok else 1
        print("%-26s first-token top-5 %s, tokens %d/%d, first difference %s" % (
            label, "identical" if same_first else "DIFFER %s / %s" % (X[x]["first_top5"][:2], Y[y]["first_top5"][:2]),
            len(gx), len(gy), "none" if div is None else div))
    print("COMPARE %s" % ("ALL IDENTICAL" if bad == 0 else "%d DIFFER" % bad))

def agents(url, out):
    pre = preamble()
    rows, lock = [], threading.Lock()
    def agent(wave, k):
        msgs = [{"role": "system", "content": pre}, {"role": "user", "content": USERS[k % len(USERS)] + f" (case {wave}-{k})"}]
        for turn in range(3):
            body = {"model": "bench", "messages": msgs, "temperature": 0.3, "max_tokens": 120, "stream": True,
                    "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            t0, first, text, err = time.time(), None, "", None
            try:
                with urllib.request.urlopen(req, timeout=1800) as r:
                    for line in r:
                        line = line.strip()
                        if not line.startswith(b"data:") or line == b"data: [DONE]": continue
                        d = json.loads(line[5:])
                        if "error" in d: err = d["error"]; break
                        for ch in d.get("choices") or []:
                            piece = (ch.get("delta") or {}).get("content")
                            if piece:
                                if first is None: first = time.time() - t0
                                text += piece
            except Exception as e:
                err = repr(e)
            with lock:
                rows.append({"wave": wave, "agent": k, "turn": turn, "ttft": first, "seconds": time.time() - t0,
                             "chars": len(text), "error": err, "head": text[:60]})
            if err: return
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "Tool result: no matching records. Continue with your next step briefly."}]
    for wave in (1, 2):
        th = []
        for k in range(4):
            t = threading.Thread(target=agent, args=(wave, k)); t.start(); th.append(t); time.sleep(1.0)
        for t in th: t.join()
    json.dump(rows, open(out, "w"), indent=1)
    for r in sorted(rows, key=lambda r: (r["wave"], r["turn"], r["agent"])):
        print("wave %d agent %d turn %d  ttft %s  total %5.1f s  %4d chars  %s" % (
            r["wave"], r["agent"], r["turn"], "%.2f s" % r["ttft"] if r["ttft"] else "-", r["seconds"], r["chars"],
            ("ERROR " + str(r["error"])[:100]) if r["error"] else repr(r["head"])))
    for wave in (1, 2):
        t0 = [r["ttft"] for r in rows if r["wave"] == wave and r["turn"] == 0 and r["ttft"]]
        print("AGENTS wave %d: first-turn TTFT %s, errors %d" % (wave, " ".join("%.2f" % x for x in sorted(t0)),
              sum(1 for r in rows if r["wave"] == wave and r["error"])))

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode in ("fork", "ref"): run_prompts(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    elif mode == "compare": compare(sys.argv[2], sys.argv[3])
    elif mode == "agents": agents(sys.argv[2], sys.argv[3])
