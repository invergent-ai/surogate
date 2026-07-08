"""Fresh-window rollout sample (steps 121-130, the repair-era mix) vs the 103-110 baseline window.
Per lane: reward mix + win rate; workflow shapes; model-slot routing. Repair lane: every group's
win structure + routing + one verbatim WINNING plan and one LOSING plan. Reads bins only."""
import ast, collections, re, sys

sys.path.insert(0, "/home/densemax/work/flavius/surogate")
import msgspec
from surogate.grpo.transport.types import TrainingBatch
from transformers import AutoTokenizer

RUN = "/home/densemax/work/flavius/surogate/output/fugu_ultra_paper"
RAW = "/var/lib/mesh/flavius/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
W = ["opus", "gemini", "gpt", "glm"]
tok = AutoTokenizer.from_pretrained(RAW)
DEC = msgspec.msgpack.Decoder(type=TrainingBatch)

def grab(text, name):
    m = re.search(rf"{name}\s*=\s*(?=\[)", text)
    if not m:
        return None
    i, depth, in_str, q = m.end(), 0, False, ""
    for j in range(i, min(len(text), i + 3000)):
        c = text[j]
        if in_str:
            if c == q and text[j - 1] != "\\":
                in_str = False
            continue
        if c in "\"'":
            in_str, q = True, c
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                try:
                    return ast.literal_eval(text[i:j + 1])
                except Exception:
                    return None
    return None

def shape(acc):
    if acc is None:
        return "unparsed"
    k = len(acc)
    def nz(a, i):
        if a in ("all", ["all"]) or (isinstance(a, list) and "all" in a):
            return set(range(i))
        return set(x for x in (a if isinstance(a, list) else [a]) if isinstance(x, int))
    A = [nz(a, i) for i, a in enumerate(acc)]
    if k == 1: return "solo"
    if k == 2: return "chain2" if A[1] else "indep2"
    if k == 3:
        if not A[1] and A[2] == {0, 1}: return "par2+agg"
        if A[1] == {0} and A[2] == {1}: return "chain3-linear"
        if A[1] == {0} and A[2] == {0, 1}: return "chain3-cumul"
        if not A[1] and not A[2]: return "indep3"
        return "other3"
    return f"{k}step"

def lane_of(prompt):
    if "PREVIOUS SOLUTION ATTEMPT" in prompt: return "repair"
    q = prompt.split("USER QUESTION:")[-1][:200]
    return "code" if "Write a complete Python program" in q else "math/reason"

def scan(steps):
    out = {"lane": collections.defaultdict(lambda: collections.Counter()),
           "shape": collections.defaultdict(lambda: collections.Counter()),
           "slots": collections.defaultdict(lambda: collections.Counter()),
           "openers": collections.defaultdict(lambda: collections.Counter())}
    repair_groups, repair_texts = collections.defaultdict(list), {"win": None, "loss": None}
    for step in steps:
        try:
            b = DEC.decode(open(f"{RUN}/run_default/rollouts/step_{step}/rollouts.bin", "rb").read())
        except Exception:
            continue
        for s in b.examples:
            prompt = tok.decode(s.prompt_ids, skip_special_tokens=True)
            lane = lane_of(prompt)
            text = tok.decode(s.completion_ids, skip_special_tokens=True)
            models = grab(text, r"model[_ ]?id")
            acc = grab(text, r"access[_ ]?list")
            r = float(s.reward)
            out["lane"][lane][r] += 1
            out["shape"][lane][shape(acc)] += 1
            if isinstance(models, list):
                for mi in models:
                    if isinstance(mi, int) and 0 <= mi <= 3:
                        out["slots"][lane][W[mi]] += 1
                if models and isinstance(models[0], int) and 0 <= models[0] <= 3:
                    out["openers"][lane][W[models[0]]] += 1
            if lane == "repair":
                gk = (step, str(s.prompt_ids[:120]))
                repair_groups[gk].append(r)
                if r >= 1.0 and repair_texts["win"] is None:
                    repair_texts["win"] = (step, text[:700])
                if r == 0.5 and repair_texts["loss"] is None:
                    repair_texts["loss"] = (step, text[:400])
    return out, repair_groups, repair_texts

fresh, rg, rt = scan(range(121, 131))
base, _, _ = scan(range(103, 111))

def winrate(cnt):
    n = sum(cnt.values())
    return (sum(v for k, v in cnt.items() if k >= 1.0) / n, n) if n else (0, 0)

print("=== win rate per lane: fresh window (121-130) vs baseline (103-110) ===")
for lane in ("code", "math/reason", "repair"):
    wf, nf = winrate(fresh["lane"][lane]); wb, nb = winrate(base["lane"][lane])
    print(f"{lane:12} fresh {wf:5.1%} (n={nf:4})   baseline {wb:5.1%} (n={nb:4})")

print("\n=== reward mix, fresh window ===")
for lane, cnt in fresh["lane"].items():
    n = sum(cnt.values())
    print(f"{lane:12} win {cnt.get(1.0,0)/n:5.1%}  valid-wrong {cnt.get(0.5,0)/n:5.1%}  parse0 {cnt.get(0.0,0)/n:5.1%}  (n={n})")

print("\n=== workflow shapes, fresh vs baseline (share of lane) ===")
for lane in ("code", "math/reason", "repair"):
    tot_f = sum(fresh["shape"][lane].values()) or 1
    tot_b = sum(base["shape"][lane].values()) or 1
    keys = set(fresh["shape"][lane]) | set(base["shape"][lane])
    rows = sorted(keys, key=lambda k: -fresh["shape"][lane][k])
    print(f"-- {lane}: " + "  ".join(f"{k} {fresh['shape'][lane][k]/tot_f:.0%} (was {base['shape'][lane][k]/tot_b:.0%})" for k in rows[:5]))

print("\n=== model usage (share of workflow slots), fresh vs baseline ===")
for lane in ("code", "math/reason", "repair"):
    tf = sum(fresh["slots"][lane].values()) or 1
    tb = sum(base["slots"][lane].values()) or 1
    print(f"-- {lane}: " + "  ".join(f"{w} {fresh['slots'][lane][w]/tf:.0%} (was {base['slots'][lane][w]/tb:.0%})" for w in W))

print("\n=== repair groups, all draws in 121-130 ===")
for (step, _), rs in sorted(rg.items()):
    c = collections.Counter(rs)
    print(f"step {step}: n={len(rs)} wins={c.get(1.0,0)} valid-wrong={c.get(0.5,0)} parse0={c.get(0.0,0)}")

for kind in ("win", "loss"):
    if rt[kind]:
        step, txt = rt[kind]
        print(f"\n=== verbatim {kind.upper()} repair plan (step {step}) ===\n{txt}")
