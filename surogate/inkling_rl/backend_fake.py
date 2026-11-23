"""A tiny CPU stand-in for Inkling with the SAME adapter layout and the same routing-replay interface, for the shim's tests.

Looped MoE: visits map to physical layers (default 0, 1, 1, 1, 2 = a prelude, a 3-pass core layer, a coda), each visit
h <- h + causal_mix(h) + moe(norm(h)); top-k routing (descending logit, ties to the lower id) with per-visit capture and
replay (ids given, gate weights from the visit's own router); experts gate_up = W13 x (+ LoRA, rows INTERLEAVED gate 2j /
up 2j+1), y = silu(gate) * up, down = W2 y (+ LoRA). LoRA per physical layer in the rig layout {A13 [E, r, H],
B13 [E, 2I, r], A2 [E, r, I], B2 [E, H, r]}, scale alpha / r = 2. fp32 on the CPU: deterministic.

Two roles:
 - "engine": sample(...) at temperature 1 with free routing and capture, using an adapter as the ENGINE folds it
   (adapter.engine_bank of the exported file's values), returning ids, log-probs and routing rows like our engine's
   return_token_ids / logprobs / routed_experts;
 - "trainer" backend for step.run_step: score / logprobs_grad with routing replayed, LoRA work values = bf16 of the
   masters (the optimizer keeps the fp32 masters; work values are what a broadcast hands the engine).
"""
from __future__ import annotations

import numpy as np
import torch

from . import adapter as AD

KEYS = AD.KEYS


class FakeInkling:
    def __init__(self, seed: int = 0, vocab: int = 64, hidden: int = 16, inter: int = 8, experts: int = 8, top_k: int = 2,
                 visit_layers=(0, 1, 1, 1, 2), rank: int = 4, alpha: int = 8):
        g = torch.Generator().manual_seed(seed)
        self.V, self.H, self.I, self.E, self.K, self.r = vocab, hidden, inter, experts, top_k, rank
        self.alpha, self.scale = alpha, alpha / rank
        self.visit_layers = tuple(visit_layers)
        self.layers = max(self.visit_layers) + 1
        self.visits, self.top_k = len(self.visit_layers), top_k
        rn = lambda *s, std=1.0: torch.randn(*s, generator=g) * std  # noqa: E731
        self.emb = rn(vocab, hidden, std=1.0)
        self.head = rn(vocab, hidden, std=0.5)
        self.router = [rn(experts, hidden, std=1.0) for _ in range(self.layers)]
        self.w13 = [rn(experts, 2 * inter, hidden, std=hidden ** -0.5) for _ in range(self.layers)]
        self.w2 = [rn(experts, hidden, inter, std=inter ** -0.5) for _ in range(self.layers)]
        self.mix = [torch.tensor(0.3 + 0.1 * i) for i in range(self.layers)]
        # trainable LoRA (work values), PEFT-like init: A uniform +-1/sqrt(fan_in), B = 0
        self.lora = []
        for _ in range(self.layers):
            A13 = (torch.rand(experts, rank, hidden, generator=g) * 2 - 1) / hidden ** 0.5
            A2 = (torch.rand(experts, rank, inter, generator=g) * 2 - 1) / inter ** 0.5
            self.lora.append({"A13": torch.nn.Parameter(A13), "B13": torch.nn.Parameter(torch.zeros(experts, 2 * inter, rank)),
                              "A2": torch.nn.Parameter(A2), "B2": torch.nn.Parameter(torch.zeros(experts, hidden, rank))})

    # ------------------------------------------------------------------ the trainer backend interface
    engine_sha = None

    def lora_layers(self) -> list:
        return self.lora

    def parameters(self) -> list:
        return [d[k] for d in self.lora for k in KEYS]

    def score(self, row, replay) -> np.ndarray:
        with torch.no_grad():
            lp, _ = self.forward(torch.as_tensor(row.ids), replay=replay, lora=self.lora)
        return lp.double().numpy()

    def logprobs_grad(self, row, replay) -> torch.Tensor:
        lp, _ = self.forward(torch.as_tensor(row.ids), replay=replay, lora=self.lora)
        return lp

    # ------------------------------------------------------------------ the model
    @staticmethod
    def _rows(W, X):
        """W @ X[t] per row: one arithmetic per token whatever T is (the property our engine guarantees)."""
        return torch.stack([W @ X[t] for t in range(X.shape[0])])

    def _moe(self, x, layer, visit, replay, lora, capture):
        logits = self._rows(self.router[layer], x)                           # [T, E]
        T = x.shape[0]
        own = self._select(logits)
        if replay is not None:
            ids = torch.as_tensor(replay.ids[:T, visit, :], dtype=torch.long)
            cov = torch.as_tensor(replay.covered[:T])
            ids = torch.where(cov[:, None], ids, own)                         # uncovered: own routing (the kit's rule)
        else:
            ids = own
        capture[:, visit, :] = ids
        out = torch.zeros_like(x)
        for t in range(T):
            g = torch.softmax(logits[t][ids[t]], dim=-1)                     # gate weights from this visit's router
            for s in range(self.K):
                e = int(ids[t, s])
                gu = self.w13[layer][e] @ x[t]
                if lora is not None:
                    gu = gu + self.scale * (lora[layer]["B13"][e] @ (lora[layer]["A13"][e] @ x[t]))
                y = torch.nn.functional.silu(gu[0::2]) * gu[1::2]
                dn = self.w2[layer][e] @ y
                if lora is not None:
                    dn = dn + self.scale * (lora[layer]["B2"][e] @ (lora[layer]["A2"][e] @ y))
                out[t] = out[t] + g[s] * dn
        return out

    def _select(self, logits):
        order = torch.argsort(-logits.detach(), dim=1, stable=True)          # ties: the lower id first
        return order[:, : self.K]

    def _trunk(self, ids, replay, lora):
        x = self.emb[ids]
        T = x.shape[0]
        capture = torch.full((T, self.visits, self.K), -1, dtype=torch.long)
        for v, layer in enumerate(self.visit_layers):
            csum = torch.cumsum(x, 0) / torch.arange(1, T + 1, dtype=x.dtype)[:, None]
            h = x + self.mix[layer] * csum
            n = h / (h.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
            x = h + self._moe(n, layer, v, replay, lora, capture)
        return x, capture

    def forward(self, ids: torch.Tensor, replay=None, lora=None):
        """log p(ids[t] | ids[:t]) for t >= 1 (0 at t = 0) and the routing capture [T, visits, K]."""
        x, capture = self._trunk(ids, replay, lora)
        T = x.shape[0]
        lp = [torch.zeros((), dtype=x.dtype)]
        for t in range(T - 1):
            lp.append(torch.log_softmax(self.head @ x[t], -1)[ids[t + 1]])
        return torch.stack(lp), capture

    # ------------------------------------------------------------------ the "engine"
    def engine_lora_from_dir(self, adapter_dir):
        """The adapter as our engine holds it after a load: the bank fold of the file's values, applied with scale 1."""
        layers, cfg = AD.import_adapter(adapter_dir, physical_layers=self.layers)
        scale = cfg["lora_alpha"] / cfg["r"]
        out = []
        for d in layers:
            if d is None:
                out.append({k: torch.zeros_like(self.lora[0][k]) for k in KEYS})
                continue
            a13, b13 = AD.engine_bank(d["A13"], d["B13"], scale)
            a2, b2 = AD.engine_bank(d["A2"], d["B2"], scale)
            out.append({"A13": torch.from_numpy(a13), "B13": torch.from_numpy(b13), "A2": torch.from_numpy(a2),
                        "B2": torch.from_numpy(b2)})
        return out, 1.0

    def sample(self, prompt: list[int], n_new: int, engine_lora, engine_scale: float, gen: torch.Generator):
        """Temperature-1 sampling with free routing; returns (token ids, their log-probs, routing rows [P+n-1, V, K])
        as our engine reports them (the last sampled token has no row)."""
        saved = self.scale
        self.scale = engine_scale
        try:
            ids = list(prompt)
            lps = []
            with torch.no_grad():
                for _ in range(n_new):
                    x = torch.as_tensor(ids)
                    # full recompute each token (tiny model): logits of the last position
                    lp_all, cap = self._full_logits(x, engine_lora)
                    probs = lp_all.exp()
                    tok = int(torch.multinomial(probs, 1, generator=gen))
                    lps.append(float(lp_all[tok]))
                    ids.append(tok)
                _, cap = self.forward(torch.as_tensor(ids[:-1]), lora=engine_lora)
            return ids[len(prompt):], lps, cap.numpy().astype(np.int16)
        finally:
            self.scale = saved

    def _full_logits(self, ids, lora):
        x, capture = self._trunk(ids, None, lora)
        return torch.log_softmax(self.head @ x[-1], -1), capture
