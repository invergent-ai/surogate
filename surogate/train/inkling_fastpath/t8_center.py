"""Lane RX: score centering for truncated sampling (coordinator's ruling 2026-11-22 08:2x; CENSUS-PAGE.md "SCORE CENTERING").
The rollouts sample from pi^S: the model's distribution cut to the support S_t (top_k 20, then top_p 0.95 on the top-20 mass, T = 1) and
renormalised. The score of what was sampled is grad log pi^S(x_t) = grad [log pi(x_t) - log sum_{v in S_t} pi(v)], S_t fixed; at the base
point its seed on the logits is e_{x_t} - q_t, q_t = pi renormalised on S_t (at most 20 non-zeros). Its mean under q_t is zero; the
uncentred seed e_x - pi has a non-zero mean at every token (it pushes the mass outside S_t into S_t).
SUPPORT RULE, read from the engine source at 22a4ca54 (M13b). The serving sampler of surogate-engine is
csrc/src/serve/ops/kernel/sampling_device.cuh, `sampling_normalize_support` (fp32: prob = expf(value / T - max) over the top-k candidates in
descending order, sum over them, the support ends at the first rank whose inclusive cumulative sum reaches 0.95f x sum); the same rule is in
csrc/src/kernels/decode_sampling.cu (double, exclusive prefix), which I read first. Rebuilt here in float64 with the exclusive prefix; the
reviewer's fp32 emulation of the serving function gave the same support on 18,000 of 18,000 rows and differed on 78 of 30 M synthetic rows.
  value = float(logit) / temperature in double (requests of the batch: temp 1.00, top_p 0.95, top_k 20, no min_p / penalties / bias);
  stable descending radix sort of all values (equal values keep ascending token id);  top_k: the first 20;  weight = exp(value - max);
  total = the sum of those 20 weights;  top_p: rank i is kept iff the sum of the weights BEFORE it is < top_p x total (so the token that
  crosses top_p is kept, and rank 0 always);  min_p 0;  the recorded log-prob is the untruncated log-softmax over the 200,058 real rows.
LOGITS: the shim's own head output (fp32, capped 30 tanh(Wh / 30)), read from the workspace of ift_head_logprob: layout [n, output_rows] at
offset 0, real rows = the first 200,058 (the rest are not vocabulary and are not in the engine's normaliser). Checked per call: the log-prob
recomputed from these logits agrees with the shim's returned value within 1e-5 (measured 1.1e-6; not bitwise, so a support boundary could
differ from the sampler's for about one token in a million; gate 1 would show a miss).
"""
import torch

import ift_backward as B

BF16, F32, F64 = torch.bfloat16, torch.float32, torch.float64
TOP_K, TOP_P = 20, 0.95
V = B.VOCAB
CHUNK = 256


def engine_logits(c, h):
    """h [n, D] bf16 contiguous -> (fp32 [n, V] view into a fresh workspace: the engine's logits, fp32 [n] the engine's log-prob of token 0)"""
    n = h.shape[0]
    R = int(c.eng.c["output_rows"])
    need = int(c.eng.lib.shim.ift_head_workspace_bytes(n))
    if need < n * R * 4:
        raise RuntimeError(f"head workspace {need} bytes < {n} x {R} x 4: the layout assumption does not hold")
    ws = torch.empty(need, dtype=torch.uint8, device=h.device)
    return ws, R


def logits_and_support(c, hidden, pos, targets):
    """hidden [T, D] bf16 (engine's final hidden), pos [n] long = positions j whose hidden predicts targets [n] (the sampled tokens).
    -> dict: row (= pos), idx [n, 20] long (token ids, engine order), q [n, 20] f64 (0 outside the nucleus), kx [n] long (rank of the
    sampled token in idx, -1 when outside the top 20), inside [n] bool (the sampled token is in S_t), size [n], lp_gap (max |lp from these
    logits - the shim's log-prob|), ties (rows with equal values among ranks 0..20)."""
    dev = hidden.device
    n = pos.numel()
    idx = torch.empty(n, TOP_K, dtype=torch.long, device=dev)
    q = torch.empty(n, TOP_K, dtype=F64, device=dev)
    kx = torch.empty(n, dtype=torch.long, device=dev)
    gap, ties = 0.0, 0
    for a in range(0, n, CHUNK):
        p = pos[a:a + CHUNK]
        m = p.numel()
        h = hidden[p].contiguous()
        t = targets[a:a + CHUNK].to(torch.int32).contiguous()
        ws, R = engine_logits(c, h)
        out = torch.empty(m, dtype=F32, device=dev)
        c.eng._check(c.eng.lib.shim.ift_head_logprob(c.eng.h, h.data_ptr(), m, t.data_ptr(), out.data_ptr(), ws.data_ptr(), ws.numel(), c.eng.stream()), "ift_head_logprob")
        torch.cuda.current_stream(dev).synchronize()
        z = ws[:m * R * 4].view(F32).view(m, R)[:, :V].double()
        mx = z.max(1, keepdim=True).values
        lp = z.gather(1, t.long()[:, None])[:, 0] - (mx[:, 0] + torch.log(torch.exp(z - mx).sum(1)))
        gap = max(gap, float((lp - out.double()).abs().max()))
        v, i = torch.topk(z, 64, dim=1)
        o1 = torch.argsort(i, dim=1, stable=True)                     # the engine's order: value descending, equal values by ascending token id
        v, i = v.gather(1, o1), i.gather(1, o1)
        o2 = torch.argsort(v, dim=1, descending=True, stable=True)
        v, i = v.gather(1, o2), i.gather(1, o2)
        if bool((v[:, TOP_K - 1] == v[:, 63]).any()):
            raise RuntimeError("44 or more equal logits at the top-20 boundary: the top-64 shortcut does not hold for this row")
        ties += int((v[:, :TOP_K] == v[:, 1:TOP_K + 1]).any(1).sum())
        v20, i20 = v[:, :TOP_K], i[:, :TOP_K]
        w = torch.exp(v20 - v20[:, :1])
        tot = w.sum(1, keepdim=True)
        keep = (torch.cumsum(w, 1) - w) < TOP_P * tot
        qq = w * keep
        idx[a:a + m] = i20
        q[a:a + m] = qq / qq.sum(1, keepdim=True)
        hit = i20 == t.long()[:, None]
        kx[a:a + m] = torch.where(hit.any(1), hit.float().argmax(1), torch.full((m,), -1, device=dev))
        del ws, z, v, i
    if gap > 1e-5:
        raise RuntimeError(f"log-prob from the workspace logits differs from the shim's by {gap:.3e} > 1e-5: the logits layout is not what was probed")
    inside = (kx >= 0) & (q.gather(1, kx.clamp(min=0)[:, None])[:, 0] > 0)
    return {"row": pos, "idx": idx, "q": q, "kx": kx, "inside": inside, "size": (q > 0).sum(1), "lp_gap": gap, "ties": ties, "targets": targets.long()}


def _rows(W, sup, a, b):
    return W.head[sup["idx"][a:b]]  # bf16 [m, 20, D]: the support's rows of the head


def head_grad_centered(W, hidden, g, sup, x_rank=None):
    """d/d hidden of sum_t g[t] * (log pi(x_t) - sum_{v in S_t} q_t(v) log pi(v)) at the base point: seed e_x - q on the capped logits
    z = 30 tanh(W h / 30). hidden [N, D] fp32 (bf16 values), g [N]. x_rank [n]: use this rank of S_t as "the sampled token" (gate 2)."""
    dh = torch.zeros(hidden.shape, dtype=F32, device=hidden.device)
    kx = sup["kx"] if x_rank is None else x_rank
    n = sup["row"].numel()
    for a in range(0, n, 1024):
        r = sup["row"][a:a + 1024]
        ga = g[r]
        if not bool((ga != 0).any()):
            continue
        Ws = _rows(W, sup, a, a + 1024).float()
        hb = hidden[r].to(BF16).float()
        th = torch.tanh(torch.einsum("mkd,md->mk", Ws, hb) / 30.0)
        s = -sup["q"][a:a + 1024].float()
        s.scatter_add_(1, kx[a:a + 1024][:, None], torch.ones(r.numel(), 1, device=s.device))
        coef = s * (1.0 - th * th) * ga[:, None]
        dh[r] = torch.einsum("mk,mkd->md", coef, Ws)
        del Ws
    return dh


def centered_logp(W, hidden_rows, sup):
    """log pi^S(x_t) with S_t fixed (sup), from the engine's final hidden at the loss positions [n, D] bf16: z_x - logsumexp_{v in S} z_v, f64."""
    out = torch.empty(sup["row"].numel(), dtype=F64, device=hidden_rows.device)
    for a in range(0, out.numel(), 1024):
        Ws = _rows(W, sup, a, a + 1024).double()
        z = 30.0 * torch.tanh(torch.einsum("mkd,md->mk", Ws, hidden_rows[a:a + 1024].double()) / 30.0)
        qa = sup["q"][a:a + 1024]
        zs = torch.where(qa > 0, z, torch.full_like(z, float("-inf")))
        out[a:a + 1024] = z.gather(1, sup["kx"][a:a + 1024][:, None])[:, 0] - torch.logsumexp(zs, 1)
        del Ws
    return out


def expected_score_check(W, hidden, sup, n_tokens=256, seed=0):
    """Gate 2: for a random hidden direction u, sum_v q_t(v) <d/dh [log pi^S(v)], u> over S_t is 0. Returns (max |sum|, max |term|) over the
    first n_tokens tokens with a support of 2 or more."""
    sel = torch.nonzero(sup["size"] >= 2)[:n_tokens, 0]
    if sel.numel() == 0:
        return 0.0, 0.0, 0
    sub = {k: (v[sel] if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == sup["row"].numel() else v) for k, v in sup.items()}
    gen = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.randn(sel.numel(), hidden.shape[1], generator=gen).to(hidden.device)
    g = torch.zeros(hidden.shape[0], dtype=F32, device=hidden.device)
    g[sub["row"]] = 1.0
    tot = torch.zeros(sel.numel(), dtype=F64, device=hidden.device)
    big = 0.0
    for k in range(TOP_K):
        on = sub["q"][:, k] > 0
        if not bool(on.any()):
            break
        dh = head_grad_centered(W, hidden, g, sub, x_rank=torch.full((sel.numel(),), k, dtype=torch.long, device=hidden.device))
        d = (dh[sub["row"]].double() * u.double()).sum(1) * on
        tot += sub["q"][:, k] * d
        big = max(big, float(d.abs().max()))
    return float(tot.abs().max()), big, int(sel.numel())
