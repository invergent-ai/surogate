#!/usr/bin/env python3
"""Heads-only supervised training of an Inkling-2 decision adapter (System 1.5).

The JointSchemaHead + FullContextEvidencePooler in torch (a port of the engine's host heads,
csrc/src/serve/targets/inkling2/impl/decision/heads.cpp), the capture files' reader, a parity check against the engine's own logits, and
supervised training of those two heads on captured hidden rows. The backbone is frozen: the rows are what the engine's decision runner gave the
heads. Capture them by serving the adapter with SUROGATE_SERVE_INKLING_DECISION_CAPTURE=<dir> and sending the labelled rows through /v1/decisions
(decisions_eval.py keeps each reply, which names its capture file). Rows: {"request": {"state", "questions"}, "target": {"answers": {name:
{"type": "choice", "choice": key} | {"type": "noul", "noul": "0" | "1"}}}}, one JSON object per line.

  heads_sft.py parity  --adapter DIR --cap DIR --replies replies.jsonl [--n 200]
  heads_sft.py train   --adapter DIR --cap DIR --replies replies.jsonl --rows decision-train.jsonl --out DIR [--epochs 12] [--lr 1e-4] [--dev-share 0.1] [--seed 20261127]
  heads_sft.py eval    --adapter DIR [--heads heads.pt] --cap DIR --replies replies.jsonl --rows decision-validation.jsonl --manifest manifest.json --split decision-validation.jsonl --out FILE
  heads_sft.py write   --adapter DIR --heads heads.pt --out DIR      a copy of the adapter directory with the trained head tensors (and exit temperatures) replaced

Option order (measured by `parity`): a choice question's options in the order of its criteria keys; a yes / no question as (false, true)."""
import argparse
import concurrent.futures as cf
import json
import math
import os
import shutil
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

EPS = float(np.finfo(np.float32).eps)       # torch nn.RMSNorm(eps=None) on fp32
TRAINED = ("heads/joint_head/", "heads/evidence_pooler/")


def read_safetensors(path, prefix="heads/"):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in head.items():
            if k == "__metadata__" or not k.startswith(prefix):
                continue
            if v["dtype"] != "F32":
                raise SystemExit(f"{k}: dtype {v['dtype']}, the heads are FP32")
            a, b = v["data_offsets"]
            f.seek(base + a)
            out[k] = torch.from_numpy(np.frombuffer(f.read(b - a), dtype="<f4").reshape(v["shape"]).copy())
    return out


def read_cap(path):
    with open(path, "rb") as f:
        if f.read(4) != b"S15C":
            raise SystemExit(f"{path}: not a capture file")
        version, n = struct.unpack("<II", f.read(8))
        meta = json.loads(f.read(n))
        D, K = int(meta["hidden"]), int(meta["options"])
        passes = []
        for p in meta["passes"]:
            core = np.frombuffer(f.read(4 * p["core_floats"]), dtype="<f4").reshape(1 + K, D)
            ex = np.frombuffer(f.read(4 * p["exit_floats"]), dtype="<f4").reshape(1 + K, D)
            pooled = np.frombuffer(f.read(4 * p["pooled_floats"]), dtype="<f4").reshape(int(p["pooled_rows"]), D)
            passes.append((core, ex, pooled))
    return meta, passes


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def mha(P, pre, q, k, v, key_mask, heads=8):
    """nn.MultiheadAttention (batch_first, packed in_proj). q [B, Lq, E], k / v [B, Lk, E]; key_mask [B, Lk] True = a real key."""
    E = q.shape[-1]
    W, b = P[pre + "in_proj_weight"], P[pre + "in_proj_bias"]
    qq = F.linear(q, W[:E], b[:E])
    kk = F.linear(k, W[E:2 * E], b[E:2 * E])
    vv = F.linear(v, W[2 * E:], b[2 * E:])
    B, Lq, Lk, hd = q.shape[0], q.shape[1], k.shape[1], E // heads
    qq = qq.view(B, Lq, heads, hd).transpose(1, 2)
    kk = kk.view(B, Lk, heads, hd).transpose(1, 2)
    vv = vv.view(B, Lk, heads, hd).transpose(1, 2)
    s = (qq @ kk.transpose(-1, -2)) / math.sqrt(hd)
    s = s.masked_fill(~key_mask[:, None, None, :], float("-inf"))
    ctx = (torch.softmax(s, -1) @ vv).transpose(1, 2).reshape(B, Lq, E)
    return F.linear(ctx, P[pre + "out_proj/weight"], P[pre + "out_proj/bias"])


def joint_logits(P, ex, ex_mask, pooled, pooled_mask):
    """ex [B, 1 + Kmax, D] (the decide row, then the option rows), ex_mask [B, 1 + Kmax]; pooled [B, Rmax, D], pooled_mask [B, Rmax] -> logits [B, Kmax] (-inf on padding)."""
    J, Pl = "heads/joint_head/", "heads/evidence_pooler/"
    up = F.silu(F.linear(rms(pooled, P[Pl + "norm/weight"]), P[Pl + "linear_up/weight"]))
    ctx_in = pooled + F.linear(up, P[Pl + "linear_down/weight"])
    x = F.linear(ex, P[J + "proj_q/weight"], P[J + "proj_q/bias"])
    te = P[J + "type_emb/weight"]
    x = x + torch.cat([te[0:1], te[1:2].expand(x.shape[1] - 1, -1)], 0)[None]
    ctx = rms(F.linear(ctx_in, P[J + "proj_ctx/weight"], P[J + "proj_ctx/bias"]), P[J + "norm1_kv/weight"])
    x = x + mha(P, J + "cross_attn/", rms(x, P[J + "norm1_q/weight"]), ctx, ctx, pooled_mask)
    s = rms(x, P[J + "norm2/weight"])
    x = x + mha(P, J + "self_attn/", s, s, s, ex_mask)
    x = x + F.linear(F.silu(F.linear(rms(x, P[J + "norm3/weight"]), P[J + "ffn_up/weight"], P[J + "ffn_up/bias"])), P[J + "ffn_down/weight"], P[J + "ffn_down/bias"])
    x = rms(x, P[J + "norm_out/weight"])
    d, o = x[:, 0], x[:, 1:]
    pq = F.linear(d, P[J + "ptr_q/weight"], P[J + "ptr_q/bias"])
    pk = F.linear(o, P[J + "ptr_k/weight"], P[J + "ptr_k/bias"])
    bias = F.linear(o, P[J + "opt_bias/weight"], P[J + "opt_bias/bias"])[..., 0]
    logits = (pk * pq[:, None]).sum(-1) / math.sqrt(x.shape[-1]) + bias
    return logits.masked_fill(~ex_mask[:, 1:], float("-inf"))


def question_order(q):
    """the option keys in the head's order"""
    if q["type"] == "choice":
        return list(q["criteria"].keys())
    if q["type"] == "noul":
        return ["0", "1"]
    raise SystemExit(f"question type {q['type']}: not in this data")


def target_index(q, t):
    order = question_order(q)
    key = t["choice"] if t["type"] == "choice" else str(t["noul"])
    return order.index(key)


def load_index(rows_file, replies_file, manifest=None, split=None):
    """-> [{"row", "q", "cap", "target", "K", "family"}] for every question with a capture file"""
    rows = [json.loads(x) for x in open(rows_file)] if rows_file else None
    fam = None
    if manifest:
        fam, i = [], 0
        for s in json.load(open(manifest))[split]["sources_in_row_order"]:
            fam += [s["file"].split("/")[0]] * s["rows"]
    out = []
    for line in open(replies_file):
        r = json.loads(line)
        if "answers" not in r or r["answers"] is None:
            continue
        for qn, ans in r["answers"].items():
            cap = (ans.get("system15") or {}).get("capture")
            if not cap:
                continue
            e = {"row": r["row"], "q": qn, "cap": cap, "family": fam[r["row"]] if fam else r.get("family")}
            if rows is not None:
                q = rows[r["row"]]["request"]["questions"][qn]
                e["target"] = target_index(q, rows[r["row"]]["target"]["answers"][qn])
                e["K"] = len(question_order(q))
            out.append(e)
    return out


class Store:
    """every capture's exit rows and pooled rows of the 5 passes in memory (fp16), padded"""

    def __init__(self, cap_dir, index, kmax=16, rmax=64, threads=16):
        n = len(index)
        meta0, p0 = read_cap(os.path.join(cap_dir, index[0]["cap"]))
        self.D, self.NP = int(meta0["hidden"]), len(p0)
        self.ex = np.zeros((n, self.NP, 1 + kmax, self.D), dtype=np.float16)
        self.pooled = np.zeros((n, self.NP, rmax, self.D), dtype=np.float16)
        self.K = np.zeros(n, dtype=np.int64)
        self.R = np.zeros(n, dtype=np.int64)
        self.loops = np.zeros(n, dtype=np.int64)
        self.engine_logits = np.full((n, self.NP, kmax), np.nan, dtype=np.float32)

        def one(i):
            meta, passes = read_cap(os.path.join(cap_dir, index[i]["cap"]))
            K = int(meta["options"])
            if len(passes) != self.NP or K > kmax:
                raise SystemExit(f"{index[i]['cap']}: {len(passes)} passes, {K} options")
            self.K[i], self.loops[i] = K, int(meta["loops_used"])
            for p, (_, ex, pooled) in enumerate(passes):
                self.ex[i, p, :1 + K] = ex
                self.pooled[i, p, :pooled.shape[0]] = pooled
                self.R[i] = pooled.shape[0]
                self.engine_logits[i, p, :K] = meta["passes"][p]["exit_logits"]

        with cf.ThreadPoolExecutor(threads) as ex:
            list(ex.map(one, range(n)))

    def batch(self, idx, dev):
        ex = torch.from_numpy(self.ex[idx]).to(dev).float()            # [B, NP, 1 + kmax, D]
        po = torch.from_numpy(self.pooled[idx]).to(dev).float()
        K = torch.from_numpy(self.K[idx]).to(dev)
        R = torch.from_numpy(self.R[idx]).to(dev)
        exm = torch.arange(ex.shape[2], device=dev)[None] < (1 + K)[:, None]
        pom = torch.arange(po.shape[2], device=dev)[None] < R[:, None]
        return ex, exm, po, pom


def forward_all(P, st, idx, dev, passes=None):
    """logits [B, NP, kmax] for the questions idx"""
    ex, exm, po, pom = st.batch(idx, dev)
    B, NP = ex.shape[0], ex.shape[1]
    sel = list(range(NP)) if passes is None else passes
    out = []
    for p in sel:
        out.append(joint_logits(P, ex[:, p], exm, po[:, p], pom))
    return torch.stack(out, 1)


def evaluate(P, st, index, dev, temps=None, bs=512):
    """accuracy per pass, and at the engine's own stopping pass (the balanced policy of the capture run)"""
    n = len(index)
    tgt = np.array([e["target"] for e in index])
    pred = np.zeros((n, st.NP), dtype=np.int64)
    nll = np.zeros((n, st.NP))
    with torch.no_grad():
        for b0 in range(0, n, bs):
            idx = np.arange(b0, min(n, b0 + bs))
            lg = forward_all(P, st, idx, dev)
            pred[idx] = lg.argmax(-1).cpu().numpy()
            t = torch.from_numpy(tgt[idx]).to(dev)
            for p in range(st.NP):
                nll[idx, p] = F.cross_entropy(lg[:, p], t, reduction="none").cpu().numpy()
    ok = pred == tgt[:, None]
    at = ok[np.arange(n), np.clip(st.loops - 1, 0, st.NP - 1)]
    out = {"n": n, "acc_per_pass": ok.mean(0).tolist(), "nll_per_pass": nll.mean(0).tolist(), "acc_at_stop": float(at.mean()), "by_question": {}, "by_family": {}}
    for q in sorted({e["q"] for e in index}):
        m = np.array([e["q"] == q for e in index])
        out["by_question"][q] = {"n": int(m.sum()), "acc_at_stop": float(at[m].mean()), "acc_last_pass": float(ok[m, -1].mean())}
    for fm in sorted({e["family"] for e in index if e.get("family")}):
        m = np.array([e.get("family") == fm for e in index])
        out["by_family"][fm] = {q: float(at[m & np.array([e["q"] == q for e in index])].mean()) for q in out["by_question"]}
    return out, ok, at


def cmd_parity(a):
    P = {k: v for k, v in read_safetensors(os.path.join(a.adapter, "decision.safetensors")).items()}
    index = load_index(None, a.replies)[:a.n]
    worst, n_p, order = 0.0, 0, {}
    replies = {}
    for line in open(a.replies):
        r = json.loads(line)
        if r.get("answers"):
            replies[r["row"]] = r["answers"]
    rows = [json.loads(x) for x in open(a.rows)] if a.rows else None
    adapter_meta = json.load(open(os.path.join(a.adapter, "decision_adapter.json")))
    for e in index:
        meta, passes = read_cap(os.path.join(a.cap, e["cap"]))
        K = int(meta["options"])
        for p, (_, ex, pooled) in enumerate(passes):
            exm = torch.ones(1, 1 + K, dtype=torch.bool)
            pom = torch.ones(1, pooled.shape[0], dtype=torch.bool)
            lg = joint_logits(P, torch.from_numpy(ex.copy())[None], exm, torch.from_numpy(pooled.copy())[None], pom)[0].numpy()
            d = float(np.abs(lg - np.array(meta["passes"][p]["exit_logits"], dtype=np.float32)).max())
            worst, n_p = max(worst, d), n_p + 1
        if rows is not None:                                   # which option each logit belongs to: the reply's probabilities against softmax(logits / T) at the stopping pass
            ans = replies[e["row"]][e["q"]]
            lu = int(meta["loops_used"])
            T = float(adapter_meta["exit_temps"][str(lu)])
            pr = torch.softmax(torch.tensor(meta["passes"][lu - 1]["exit_logits"], dtype=torch.float64) / T, -1).numpy()
            q = rows[e["row"]]["request"]["questions"][e["q"]]
            if q["type"] == "choice":
                got = [ans["probabilities"][k] for k in question_order(q)]
                order.setdefault("choice_in_criteria_order", []).append(float(np.abs(np.array(got) - pr).max()))
            else:
                order.setdefault("noul_true_is_index_1", []).append(abs(ans["noul"] - pr[1]))
                order.setdefault("noul_true_is_index_0", []).append(abs(ans["noul"] - pr[0]))
    print(json.dumps({"questions": len(index), "passes": n_p, "max_abs_logit_difference": worst,
                      "order_check_max_abs": {k: max(v) for k, v in order.items()}, "order_check_mean_abs": {k: float(np.mean(v)) for k, v in order.items()}}))


def cmd_train(a):
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    P0 = read_safetensors(os.path.join(a.adapter, "decision.safetensors"))
    index = load_index(a.rows, a.replies)
    rows_all = sorted({e["row"] for e in index})
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(rows_all))
    dev_rows = {rows_all[i] for i in perm[:int(len(rows_all) * a.dev_share)]}
    t0 = time.time()
    st = Store(a.cap, index)
    print(f"loaded {len(index)} questions in {time.time() - t0:.0f} s; rows {len(rows_all)}, dev rows {len(dev_rows)}", flush=True)
    tr_idx = np.array([i for i, e in enumerate(index) if e["row"] not in dev_rows])
    dv_idx = np.array([i for i, e in enumerate(index) if e["row"] in dev_rows])
    dv_index = [index[i] for i in dv_idx]

    class View:                                               # the dev questions as their own store
        pass
    dv = View()
    for k in ("ex", "pooled", "K", "R", "loops", "engine_logits"):
        setattr(dv, k, getattr(st, k)[dv_idx])
    dv.D, dv.NP = st.D, st.NP
    dv.batch = lambda idx, d: Store.batch(dv, idx, d)
    P = {k: (v.clone().to(dev).requires_grad_(k.startswith(TRAINED))) for k, v in P0.items()}
    params = [v for k, v in P.items() if v.requires_grad]
    print(f"trained tensors {len(params)}, parameters {sum(p.numel() for p in params) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    tgt = torch.tensor([e["target"] for e in index])
    base, _, _ = evaluate(P, dv, dv_index, dev)
    print("dev, heads as shipped:", json.dumps({k: base[k] for k in ("n", "acc_per_pass", "acc_at_stop", "by_question")}), flush=True)
    os.makedirs(a.out, exist_ok=True)
    hist, best = [{"epoch": 0, **{k: base[k] for k in ("acc_per_pass", "nll_per_pass", "acc_at_stop")}}], (base["acc_at_stop"], 0)
    steps = a.epochs * math.ceil(len(tr_idx) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05, anneal_strategy="cos")
    for ep in range(1, a.epochs + 1):
        order = tr_idx[rng.permutation(len(tr_idx))]
        tot, nb, t1 = 0.0, 0, time.time()
        for b0 in range(0, len(order), a.batch):
            idx = np.sort(order[b0:b0 + a.batch])
            lg = forward_all(P, st, idx, dev)                 # [B, NP, kmax]
            t = tgt[idx].to(dev)
            loss = sum(F.cross_entropy(lg[:, p], t) for p in range(st.NP)) / st.NP
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tot, nb = tot + float(loss), nb + 1
        ev, _, _ = evaluate(P, dv, dv_index, dev)
        rec = {"epoch": ep, "train_loss": tot / nb, "acc_per_pass": ev["acc_per_pass"], "nll_per_pass": ev["nll_per_pass"], "acc_at_stop": ev["acc_at_stop"], "seconds": time.time() - t1}
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        if ev["acc_at_stop"] > best[0]:
            best = (ev["acc_at_stop"], ep)
            torch.save({k: v.detach().cpu() for k, v in P.items()}, os.path.join(a.out, "heads.pt"))
    json.dump({"args": vars(a), "history": hist, "best_dev_acc_at_stop": best[0], "best_epoch": best[1], "dev_rows": len(dev_rows), "train_questions": int(len(tr_idx))},
              open(os.path.join(a.out, "train.json"), "w"), indent=1)
    print(f"best dev accuracy at the stopping pass {best[0]:.4f} at epoch {best[1]} (as shipped {base['acc_at_stop']:.4f})", flush=True)


def cmd_eval(a):
    dev = torch.device("cuda")
    P = read_safetensors(os.path.join(a.adapter, "decision.safetensors"))
    if a.heads:
        P.update(torch.load(a.heads))
    P = {k: v.to(dev) for k, v in P.items()}
    index = load_index(a.rows, a.replies, a.manifest, a.split)
    st = Store(a.cap, index)
    ev, _, _ = evaluate(P, st, index, dev)
    json.dump(ev, open(a.out, "w"), indent=1)
    print(json.dumps({k: ev[k] for k in ("n", "acc_per_pass", "acc_at_stop", "by_question")}))


def cmd_write(a):
    new = torch.load(a.heads)
    src = os.path.join(a.adapter, "decision.safetensors")
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.adapter):
        if f != "decision.safetensors":
            shutil.copy2(os.path.join(a.adapter, f), os.path.join(a.out, f))
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head_raw = f.read(n)
        head = json.loads(head_raw)
        body = bytearray(f.read())
    changed = 0
    for k, v in head.items():
        if k == "__metadata__" or not k.startswith(TRAINED):
            continue
        t = new[k].to(torch.float32).contiguous().numpy()
        if list(t.shape) != v["shape"]:
            raise SystemExit(f"{k}: shape {list(t.shape)} != {v['shape']}")
        a0, b0 = v["data_offsets"]
        body[a0:b0] = t.astype("<f4").tobytes()
        changed += 1
    with open(os.path.join(a.out, "decision.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", n))
        f.write(head_raw)
        f.write(body)
    print(json.dumps({"tensors_replaced": changed, "out": a.out}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("parity", "train", "eval", "write"))
    ap.add_argument("--adapter", required=True); ap.add_argument("--cap", default=""); ap.add_argument("--replies", default=""); ap.add_argument("--rows", default="")
    ap.add_argument("--manifest", default=""); ap.add_argument("--split", default=""); ap.add_argument("--heads", default=""); ap.add_argument("--out", default="")
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01); ap.add_argument("--batch", type=int, default=128); ap.add_argument("--dev-share", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20261127)
    a = ap.parse_args()
    {"parity": cmd_parity, "train": cmd_train, "eval": cmd_eval, "write": cmd_write}[a.cmd](a)


if __name__ == "__main__":
    main()
