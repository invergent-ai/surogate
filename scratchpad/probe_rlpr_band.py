"""RLPR difficulty-band probe: do RLPR items land in the LEARNABLE band at our worker
handicap (4096 tok / temp 0.2 / minimal)? Samples 40 items (stratified over categories,
skipping trivially-short golds), runs all 4 workers solo, grades with our math grader
(normalizer fallback), and reports per-worker rates + per-item solver counts.
Dead-easy = 4/4 workers solve; fortress = 0/4; contested = 1-3/4 (the gradient band)."""
import asyncio, importlib.util, itertools, json, random, sys

sys.path.insert(0, "/home/densemax/work/flavius/surogate/ultra")
sys.path.insert(0, "/home/densemax/work/flavius/surogate")
spec = importlib.util.spec_from_file_location(
    "fpe", "/home/densemax/work/flavius/surogate/environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
from ultra.workers import Sampling
from ultra.grading import get_grader
from datasets import load_dataset

D = "/home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/grpo_pilot_train"
cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
WORKERS = cfg["worker_pool_names"]

print("sampling RLPR items (streaming)...", flush=True)
ds = load_dataset("openbmb/RLPR-Train-Dataset", split="train", streaming=True)
pool_items = []
for r in itertools.islice(ds, 800):
    gold = str(r["reward_model"].get("ground_truth", "")).strip()
    q = next((m["content"] for m in r["prompt"] if m["role"] == "user"), "")
    at = r.get("extra_info", {}).get("answer_type", "")
    if not gold or not q or len(q) < 80 or len(gold) > 60:
        continue
    if at in ("Boolean",):   # trivially guessable
        continue
    pool_items.append({"q": q, "gold": gold, "cat": r.get("ability", "?"), "at": at,
                       "diff": r.get("extra_info", {}).get("difficulty", "?")})
rng = random.Random(11)
rng.shuffle(pool_items)
by_cat = {}
sample = []
for it in pool_items:           # stratify: max 5 per category, 40 total
    if by_cat.get(it["cat"], 0) >= 5:
        continue
    by_cat[it["cat"]] = by_cat.get(it["cat"], 0) + 1
    sample.append(it)
    if len(sample) == 40:
        break
print(f"sample: {len(sample)} items across {len(by_cat)} categories: {by_cat}", flush=True)

pool = env._build_pool(pilot_config=cfg, provider_mode="live",
                       cache_dir=".ultra_cache/rlpr_probe", max_concurrency=4,
                       requests_per_minute=None, timeout_s=300.0, max_retries=2)
samp = Sampling(temperature=0.2, top_p=1.0, max_tokens=4096, reasoning_effort="minimal")
grader = get_grader("math_equal")

async def solo(it, w):
    msgs = [{"role": "user", "content": it["q"] + "\n\nProvide only the final answer in \\boxed{} format."}]
    try:
        c = await pool.call(w, msgs, samp)
        score = await asyncio.wait_for(asyncio.to_thread(grader, c.text or "", it["gold"]), timeout=60.0)
        return float(score >= 0.5)
    except Exception:
        return 0.0

async def main():
    res = await asyncio.gather(*[solo(it, w) for it in sample for w in WORKERS])
    k = 0
    per_worker = {w: 0 for w in WORKERS}
    buckets = {"dead-easy(4/4)": 0, "contested(1-3)": 0, "fortress(0/4)": 0}
    contested_examples = []
    for it in sample:
        solvers = 0
        for w in WORKERS:
            per_worker[w] += res[k]
            solvers += int(res[k]); k += 1
        b = "dead-easy(4/4)" if solvers == 4 else ("fortress(0/4)" if solvers == 0 else "contested(1-3)")
        buckets[b] += 1
        if b == "contested(1-3)" and len(contested_examples) < 3:
            contested_examples.append((solvers, it["cat"], it["q"][:90]))
    n = len(sample)
    print("\n==================== RLPR BAND VERDICT ====================")
    print("per-worker solo rates:", {w: round(per_worker[w] / n, 2) for w in WORKERS})
    for b, c in buckets.items():
        print(f"  {b}: {c}/{n} ({c/n:.0%})")
    print("\ncontested examples:")
    for s, cat, q in contested_examples:
        print(f"  [{s}/4 solved] ({cat}) {q}...")
    print("\nBAND FIT:", "GOOD - majority contested" if buckets["contested(1-3)"] >= n * 0.4 else
          ("TOO EASY for this pool" if buckets["dead-easy(4/4)"] > n * 0.5 else "TOO HARD at handicap"))

asyncio.run(main())
