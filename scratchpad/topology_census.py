"""ACCESS-TOPOLOGY census over fresh training rollouts (row-110 checkup instrument).

Classifies each rollout's access_list into a workflow SHAPE and reports, per lane:
share + win rate per shape. Shapes: solo / chain2 / indep2 / chain3-linear /
chain3-cumulative / par2+aggregator (the isolation-exploiting fan-in) / indep3 / other.
Lane attribution: exact task-text match against the three lane manifests (not a heuristic).
Reads rollouts.bin only — zero interaction with the live run.
"""
import collections, json, re, sys

sys.path.insert(0, "/home/densemax/work/flavius/surogate")
import msgspec
from surogate.grpo.transport.types import TrainingBatch
from transformers import AutoTokenizer

RUN = "/home/densemax/work/flavius/surogate/output/fugu_ultra_paper"
RAW = "/var/lib/mesh/flavius/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
D = "/home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/grpo_pilot_train"
STEPS = list(range(103, 111))
tok = AutoTokenizer.from_pretrained(RAW)
DEC = msgspec.msgpack.Decoder(type=TrainingBatch)

def norm(s):
    return re.sub(r"\s+", " ", s).strip()[:80]

# the prompt embeds the task's SYSTEM message at the head of USER QUESTION ("[system] ...");
# each lane has ONE distinct system text, so the normalized system prefix identifies the lane.
lane_by_key = {}
for lane, fn in [("math", "hard_mix_math_taskspecs.jsonl"), ("code", "hard_mix_code_taskspecs.jsonl"),
                 ("reason", "hard_mix_rlpr_taskspecs.jsonl")]:
    for l in open(f"{D}/{fn}"):
        d = json.loads(l)
        s = " ".join(m.get("content", "") for m in d["input"]["messages"] if m.get("role") == "system")
        if s.strip():
            lane_by_key[norm("[system] " + s)] = lane
        else:  # no system message: fall back to user-text prefix
            u = " ".join(m.get("content", "") for m in d["input"]["messages"] if m.get("role") == "user")
            lane_by_key[norm(u)] = lane

def grab_access(text):
    m = re.search(r"access[_ ]?list\s*=\s*(?=\[)", text)
    if not m:
        return None
    i, depth, in_str, q = m.end(), 0, False, ""
    for j in range(i, min(len(text), i + 2000)):
        c = text[j]
        if in_str:
            if c == q and text[j-1] != "\\":
                in_str = False
            continue
        if c in "\"'":
            in_str, q = True, c
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                import ast
                try:
                    return ast.literal_eval(text[i:j+1])
                except Exception:
                    return None
    return None

def shape(acc):
    if acc is None:
        return "unparsed"
    k = len(acc)
    def nz(a, i):   # normalize: "all" -> all prior indices
        if a in ("all", ["all"]) or (isinstance(a, list) and "all" in a):
            return set(range(i))
        return set(x for x in (a if isinstance(a, list) else [a]) if isinstance(x, int))
    A = [nz(a, i) for i, a in enumerate(acc)]
    if k == 1:
        return "solo"
    if k == 2:
        return "chain2" if A[1] else "indep2"
    if k == 3:
        a1, a2 = A[1], A[2]
        if not a1 and a2 == {0, 1}:
            return "par2+agg"
        if a1 == {0} and a2 == {1}:
            return "chain3-linear"
        if a1 == {0} and a2 == {0, 1}:
            return "chain3-cumul"
        if not a1 and not a2:
            return "indep3"
        return "other3"
    return f"{k}step"

stats = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))  # lane -> shape -> [n, wins]
step_seen = []
for step in STEPS:
    try:
        b = DEC.decode(open(f"{RUN}/run_default/rollouts/step_{step}/rollouts.bin", "rb").read())
    except Exception:
        continue
    step_seen.append(step)
    for s in b.examples:
        prompt = tok.decode(s.prompt_ids, skip_special_tokens=True)
        q = prompt.split("USER QUESTION:")[-1].split("AVAILABLE LANGUAGE MODELS")[0]
        lane = lane_by_key.get(norm(q), "unknown")
        text = tok.decode(s.completion_ids, skip_special_tokens=True)
        sh = shape(grab_access(text))
        cell = stats[lane][sh]
        cell[0] += 1
        cell[1] += int(s.reward >= 1.0)

print(f"steps read: {step_seen}\n")
print(f"{'lane':8} {'shape':14} {'n':>5} {'share':>6} {'win':>6}")
for lane in ("code", "math", "reason", "unknown"):
    tot = sum(n for n, _ in stats[lane].values())
    if not tot:
        continue
    for sh, (n, w) in sorted(stats[lane].items(), key=lambda kv: -kv[1][0]):
        print(f"{lane:8} {sh:14} {n:5} {n/tot:6.1%} {w/n:6.1%}")
    print(f"{lane:8} {'TOTAL':14} {tot:5} {'':6} {sum(w for _,w in stats[lane].values())/tot:6.1%}\n")
