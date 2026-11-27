#!/usr/bin/env python3
"""The secret-rule game for rl_secret_rule.py. Script-generated only: no download, no benchmark, no model-written text.
A card has four invented attributes; its house (1-4) is a fixed secret permutation of ONE of them (the glyph). The prompt never states the rule.
  rl_secret_rule_gen.py <out dir> [seed]     writes problems.jsonl, rule.json and split/{train.jsonl, heldout.jsonl, schedule.json}
train 4,096 prompts in blocks of 32 (step v of the run trains on block v), heldout 512; the two are disjoint."""
import hashlib, itertools, json, os, random, sys

out = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261127
rng = random.Random(seed)
GLYPH = ["kor", "vex", "mip", "tau"]
TINT = ["vesh", "olm", "drae", "quin", "sarb", "yute"]
MARK = ["dulo", "fen", "harg", "issa", "nop"]
perm = [1, 2, 3, 4]
rng.shuffle(perm)
house = dict(zip(GLYPH, perm))
TEXT = ("Zorblat cards are sorted into four houses, numbered 1 to 4, by a rule you have not been told.\n\n"
        "Card: glyph {g}, tint {t}, rank {r}, mark {m}.\n\n"
        "Which house does this card belong to? Reply with the house number only: one digit, 1, 2, 3 or 4, and nothing else.")
rows = []
for g, t, r, m in itertools.product(GLYPH, TINT, range(1, 100), MARK):
    p = TEXT.format(g=g, t=t, r=r, m=m)
    rows.append({"id": "p" + hashlib.sha256(p.encode()).hexdigest()[:16], "prompt": p, "gold": str(house[g])})
rng.shuffle(rows)
os.makedirs(os.path.join(out, "split"), exist_ok=True)
TRAIN, HELD, BLOCK = 4096, 512, 32
train, held = rows[:TRAIN], rows[TRAIN:TRAIN + HELD]


def dump(path, xs):
    with open(path, "w") as f:
        for x in xs:
            f.write(json.dumps(x, sort_keys=True) + "\n")


dump(os.path.join(out, "problems.jsonl"), rows)
dump(os.path.join(out, "split", "train.jsonl"), train)
dump(os.path.join(out, "split", "heldout.jsonl"), held)
json.dump({"seed": seed, "rule": "step v of the run trains on blocks[v]; the blocks are train.jsonl in its order", "block_size": BLOCK,
           "blocks": [[x["id"] for x in train[i:i + BLOCK]] for i in range(0, TRAIN, BLOCK)]}, open(os.path.join(out, "split", "schedule.json"), "w"))
json.dump({"seed": seed, "rule": "house = permutation of the glyph", "house_of_glyph": house, "problems": len(rows)}, open(os.path.join(out, "rule.json"), "w"), indent=1)
print(len(rows), "problems;", len(train), "train,", len(held), "held out")
