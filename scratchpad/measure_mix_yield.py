"""Signal-yield-per-subject audit: over recent steps, what fraction of draws are code vs math,
and what fraction of LEARNABLE groups (mixed outcomes, 1..63 wins) each subject delivers.
Read-only over rollouts.bin; group-level only (no completion decode)."""
import collections, hashlib, sys

sys.path.insert(0, "/home/densemax/work/flavius/surogate")
import msgspec
from surogate.grpo.transport.types import TrainingBatch
from transformers import AutoTokenizer

RUN = "/home/densemax/work/flavius/surogate/output/fugu_ultra_paper"
RAW = "/var/lib/mesh/flavius/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
tok = AutoTokenizer.from_pretrained(RAW)
DEC = msgspec.msgpack.Decoder(type=TrainingBatch)

WINDOWS = {"pre-fix 45-63": range(45, 64), "post-fix 64-70": range(64, 71)}
for label, steps in WINDOWS.items():
    stats = {k: collections.Counter() for k in ("math", "code")}
    for step in steps:
        try:
            b = DEC.decode(open(f"{RUN}/run_default/rollouts/step_{step}/rollouts.bin", "rb").read())
        except Exception:
            continue
        groups = collections.defaultdict(list)
        for s in b.examples:
            groups[hashlib.sha1(bytes(str(s.prompt_ids), "utf8")).hexdigest()[:10]].append(s)
        for gh, ss in groups.items():
            q = tok.decode(ss[0].prompt_ids, skip_special_tokens=True).split("USER QUESTION:")[-1]
            kind = "code" if ("stdin" in q[:300] or "code block" in q[:300]) else "math"
            wins = sum(1 for x in ss if x.reward >= 0.99)
            n = len(ss)
            stats[kind]["groups"] += 1
            if wins == 0:
                stats[kind]["fortress"] += 1
            elif wins >= n - 1:
                stats[kind]["saturated"] += 1
            else:
                stats[kind]["contested"] += 1
                # signal mass ~ variance of the reward vector (what the gradient sees)
                rs = [x.reward for x in ss]
                m = sum(rs) / n
                stats[kind]["var_milli"] += int(1000 * sum((r - m) ** 2 for r in rs) / n)

    total_groups = sum(stats[k]["groups"] for k in stats)
    total_contested = sum(stats[k]["contested"] for k in stats)
    total_var = sum(stats[k]["var_milli"] for k in stats)
    print(f"=== {label} (n={total_groups} groups) ===")
    for k in ("math", "code"):
        g = stats[k]
        print(f"  {k}: draws {g['groups']}/{total_groups} ({g['groups']/max(total_groups,1):.0%})"
              f" | contested {g['contested']} ({g['contested']/max(g['groups'],1):.0%} of its draws)"
              f" | fortress {g['fortress']} | near-saturated {g['saturated']}"
              f" | share of ALL contested groups: {g['contested']/max(total_contested,1):.0%}"
              f" | share of gradient mass: {g['var_milli']/max(total_var,1):.0%}")
    print()

# pool composition for reference
import json
kinds = collections.Counter()
for line in open("/home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/grpo_pilot_train/hard_mix_all_taskspecs.jsonl"):
    d = json.loads(line)
    src = str(d.get("source", d.get("dataset", "?")))
    kinds[src] += 1
print("pool composition by source:", dict(kinds))
