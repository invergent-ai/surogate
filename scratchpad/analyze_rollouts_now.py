"""Post-intervention rollout census (steps 65-70, lr 1e-5 + eviction live) vs the 53-63 baseline.
Reads rollouts.bin only. Reports: reward structure, workflow-shape census (steps_dist, chain
classes, prose-plan prevalence), per-source routing, and verbatim winner/loser from the most
contested recent group."""
import ast, collections, hashlib, re, sys

sys.path.insert(0, "/home/densemax/work/flavius/surogate")
import msgspec
from surogate.grpo.transport.types import TrainingBatch
from transformers import AutoTokenizer

RUN = "/home/densemax/work/flavius/surogate/output/fugu_ultra_paper"
RAW = "/var/lib/mesh/flavius/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
STEPS = [87, 88, 89, 90]
tok = AutoTokenizer.from_pretrained(RAW)
DEC = msgspec.msgpack.Decoder(type=TrainingBatch)


def grab_list(text, name):
    m = re.search(rf"{name}\s*=\s*(?=\[)", text)
    if not m:
        return None
    i, depth, in_str, esc, q = m.end(), 0, False, False, ""
    for j in range(i, min(len(text), i + 4000)):
        c = text[j]
        if in_str:
            if esc: esc = False
            elif c == "\\": esc = True
            elif c == q: in_str = False
            continue
        if c in "\"'": in_str, q = True, c
        elif c == "[": depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                try:
                    return ast.literal_eval(text[i:j + 1])
                except Exception:
                    return None
    return None


def classify(models, subtasks, prose_len):
    """Chain classes matching the 53-63 census definitions."""
    if models is None or subtasks is None:
        return "unparsed"
    sub0 = (subtasks[0] if subtasks else "").lower()
    subs = " ".join(str(s).lower() for s in subtasks)
    n = len(models)
    if prose_len > 200:
        return "prose-plan"
    if n == 1:
        return "1-step"
    if re.search(r"\b(solve|answer|compute|find the)\b", sub0) and n >= 2:
        return "solve-first"
    if n >= 4 and re.search(r"verif|check|review", subs) and re.search(r"format|present|final answer|box", subs):
        return "verify+format-4"
    return f"{n}-step-other"


groups_all = {}   # (step, ghash) -> list of (reward, models, subtasks, chain, completion_text)
kind_by_group = {}
for step in STEPS:
    try:
        b = DEC.decode(open(f"{RUN}/run_default/rollouts/step_{step}/rollouts.bin", "rb").read())
    except Exception:
        continue
    per_group = collections.defaultdict(list)
    for s in b.examples:
        per_group[hashlib.sha1(bytes(str(s.prompt_ids), "utf8")).hexdigest()[:10]].append(s)
    for gh, ss in per_group.items():
        q = tok.decode(ss[0].prompt_ids, skip_special_tokens=True).split("USER QUESTION:")[-1]
        kind = "code" if ("stdin" in q[:300] or "code block" in q[:300]) else "math"
        kind_by_group[(step, gh)] = kind
        rows = []
        for s in ss:
            text = tok.decode(s.completion_ids, skip_special_tokens=True)
            models = grab_list(text, r"model[_ ]?id")
            subtasks = grab_list(text, r"subtasks?")
            mpos = re.search(r"model[_ ]?id\s*=", text)
            prose_len = len(text[:mpos.start()].strip()) if mpos else len(text)
            chain = classify(models, subtasks, prose_len)
            rows.append((s.reward, models, subtasks, chain, text))
        groups_all[(step, gh)] = rows

# ---- 1. reward structure ----
print("=== reward structure per step (post-intervention window) ===")
for step in STEPS:
    gs = [(k, v) for k, v in groups_all.items() if k[0] == step]
    if not gs:
        continue
    parts = []
    for (st, gh), rows in gs:
        wins = sum(1 for r in rows if r[0] >= 0.99)
        parts.append(f"{kind_by_group[(st, gh)][0]}:{wins}/{len(rows)}")
    rs = [r[0] for _, rows in gs for r in rows]
    print(f"  step {step}: mean={sum(rs)/len(rs):.3f}  groups[{' '.join(parts)}]  (kind:wins/n)")

# ---- 2. chain census vs 53-63 baseline ----
rows_flat = [r for rows in groups_all.values() for r in rows]
print(f"\n=== workflow-shape census, steps 65-70 (n={len(rows_flat)}) ===")
steps_dist = collections.Counter(len(r[1]) for r in rows_flat if r[1])
tot = sum(steps_dist.values())
print("  steps_dist:", {k: f"{v/tot:.0%}" for k, v in sorted(steps_dist.items())},
      f" mean={sum(k*v for k,v in steps_dist.items())/tot:.2f}")
chains = collections.Counter(r[3] for r in rows_flat)
wins_by_chain = collections.defaultdict(lambda: [0, 0])
for r in rows_flat:
    wins_by_chain[r[3]][0] += (r[0] >= 0.99)
    wins_by_chain[r[3]][1] += 1
print("  chain classes (share, win-rate):")
for c, n in chains.most_common():
    w, t = wins_by_chain[c]
    print(f"    {c:<18} {n/len(rows_flat):>5.1%}  win {w}/{t} ({w/max(t,1):.0%})")
print("  [53-63 baseline: solve-first 13-17% win & halving; verify+format-4 ~22% share; prose-plan 2/2560]")

# ---- 3. routing (ordinals 0=opus 1=gemini 2=gpt 3=glm) ----
NAMES = {0: "opus", 1: "gemini", 2: "gpt", 3: "glm"}
print("\n=== routing: model slots by task kind ===")
for kind in ("math", "code"):
    slot = collections.Counter()
    first = collections.Counter()
    n_wf = 0
    for k, rows in groups_all.items():
        if kind_by_group[k] != kind:
            continue
        for r in rows:
            if not r[1]:
                continue
            n_wf += 1
            ms = [m for m in r[1] if isinstance(m, int) and 0 <= m <= 3]
            slot.update(ms)
            if ms:
                first[ms[0]] += 1
    t = sum(slot.values())
    print(f"  {kind}: all-slots " + " ".join(f"{NAMES[m]}={slot[m]/t:.0%}" for m in sorted(slot)) +
          "  |  first-step " + " ".join(f"{NAMES[m]}={first[m]/max(sum(first.values()),1):.0%}" for m in sorted(first)))

# ---- 4. verbatim winner vs loser from the most contested recent group ----
contested = [(k, rows) for k, rows in groups_all.items()
             if 0 < sum(r[0] >= 0.99 for r in rows) < len(rows)]
contested.sort(key=lambda kv: (kv[0][0], -min(sum(r[0] >= 0.99 for r in kv[1]),
                                              len(kv[1]) - sum(r[0] >= 0.99 for r in kv[1]))))
if contested:
    (st, gh), rows = contested[-1]
    wins = sum(r[0] >= 0.99 for r in rows)
    print(f"\n=== verbatim: most contested late group (step {st}, {kind_by_group[(st, gh)]}, {wins}/{len(rows)} wins) ===")
    winner = next(r for r in rows if r[0] >= 0.99)
    loser = next((r for r in rows if 0.4 < r[0] < 0.99), next(r for r in rows if r[0] < 0.99))
    for tag, r in (("WINNER", winner), ("LOSER", loser)):
        print(f"--- {tag} (reward {r[0]}) ---")
        print(r[4][:900])
        print()
