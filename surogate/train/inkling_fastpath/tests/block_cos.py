"""Do the gradients of disjoint blocks of tasks agree? The read on the full gradients (no card): grad-b<j>.pt of the blocks, as block_grads.py writes them (dict "<layer>.<key>" -> fp32).
Per tensor group and over all live tensors as one vector: the 6 pair cosines, the 4 leave-one-block-out cosines cos(G_b, sum of the others),
the norms, and from them the share r = |s|^2 / |n|^2 of a 20-world block under G_b = s + n_b (r_block = mean pair cosine / (1 - mean pair cosine)).
  block_cos.py <out.json> <grad-b0.pt> <grad-b1.pt> <grad-b2.pt> <grad-b3.pt>"""
import itertools, json, sys
import torch

out, files = sys.argv[1], sys.argv[2:]
G = [torch.load(f, map_location="cpu") for f in files]
names = list(G[0].keys())
assert all(list(g.keys()) == names for g in G)
EXPERT = ("A13", "B13", "A2", "B2")


def group_of(n):
    L, k = n.split(".")
    L = int(L)
    depth = "before" if L < 6 else ("inside" if L < 18 else "after")
    return ("exp_" if k in EXPERT else "attn_") + depth, k


def stats(sel):
    nb = len(G)
    dot = [[0.0] * nb for _ in range(nb)]
    for n in sel:
        v = [g[n].double().flatten() for g in G]
        for i in range(nb):
            for j in range(i, nb):
                d = float(torch.dot(v[i], v[j]))
                dot[i][j] += d
                if i != j:
                    dot[j][i] += d
    norm = [dot[i][i] ** 0.5 for i in range(nb)]
    if min(norm) == 0:
        return None
    pair = {f"{i}-{j}": dot[i][j] / (norm[i] * norm[j]) for i, j in itertools.combinations(range(nb), 2)}
    loo = {}
    for b in range(nb):
        o = [j for j in range(nb) if j != b]
        d = sum(dot[b][j] for j in o)
        no = sum(dot[i][j] for i in o for j in o) ** 0.5
        loo[str(b)] = d / (norm[b] * no)
    mp = sum(pair.values()) / len(pair)
    return {"norms": norm, "pair_cos": pair, "pair_mean": mp, "pair_min": min(pair.values()), "loo_cos": loo, "loo_mean": sum(loo.values()) / nb,
            "r_block": (mp / (1 - mp)) if mp < 1 else None, "coords": int(sum(G[0][n].numel() for n in sel))}


live = [n for n in names if any(bool(g[n].ne(0).any()) for g in G)]
res = {"files": files, "tensors": len(names), "live": len(live), "all_live": stats(live), "groups": {}, "keys": {}}
gs, ks = {}, {}
for n in live:
    g, k = group_of(n)
    gs.setdefault(g, []).append(n)
    ks.setdefault(k, []).append(n)
for g, sel in sorted(gs.items()):
    res["groups"][g] = stats(sel)
for k, sel in sorted(ks.items()):
    res["keys"][k] = stats(sel)
json.dump(res, open(out, "w"), indent=1)


def line(name, s):
    return f"{name:14s} coords {s['coords']:>11d}  pair mean {s['pair_mean']:+.4f} min {s['pair_min']:+.4f}  one-vs-three mean {s['loo_mean']:+.4f}  r(20 worlds) {s['r_block']:.4f}  pairs " + " ".join(f"{v:+.3f}" for v in s["pair_cos"].values())


print(f"tensors {len(names)}, live {len(live)}")
print(line("ALL LIVE", res["all_live"]))
for g, s in res["groups"].items():
    print(line(g, s))
for k, s in res["keys"].items():
    print(line("key " + k, s))
