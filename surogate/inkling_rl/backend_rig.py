"""Backend on lane L's learnability rig (lanes/L/rig/rig.py, READ ONLY: imported by path, never changed): the INTERIM
trainer for the plumbing step P4. Not the engine's forward: its step-0 engine_mismatch_kl is the independent-kernel floor
(~1e-3..1e-2 by population), above the kit's gated-start bar 0.007, so rig steps are plumbing only (agent-EN 17:2x).

Forward modes (config inkling_rl.rig.score_mode): "w4a16" = the rig's train arithmetic (act quant off, routing replayed;
the gradient pass always uses it, straight-through the quantisers); "w4a4" = the rig's eval arithmetic (act quant on)
with the engine's routing replayed, for the step-start score / pi_old / the mismatch metric (lever T4 in NOTES).
Rows: one sequence per forward (B = 1, no KV carry), so rows longer than the card fits are refused by the shim's
inkling_rl.max_row_tokens (default 2,048) and counted, not cut.
Uncovered interior routing rows cannot be mixed with own routing here: such rows are refused (replay_mixes_own False).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from . import adapter as AD

RIG_DIR = "/mnt/data/work/rl/engine-inkling-2026-11-21/lanes/L/rig"


class RigBackend:
    replay_mixes_own = False
    engine_sha = None

    def __init__(self, cfg: dict):
        ir = cfg.get("inkling_rl") or {}
        rc = ir.get("rig") or {}
        sys.path.insert(0, rc.get("rig_dir", RIG_DIR))
        import rig  # noqa: E402  (lane L's module; sets torch's default device)

        self.rig = rig
        self.lora = rig.make_lora(seed=int(rc.get("lora_seed", 0)))
        host = tuple(int(x) for x in str(rc.get("host_layers", "20,21,22,23")).split(",") if x != "")
        self.model = rig.RigModel(rc.get("model_dir", "/var/lib/effrl/inkling/model"), host_layers=host, lora=self.lora)
        self.score_mode = rc.get("score_mode", "w4a16")
        self.visits, self.top_k = 48, 8
        if int(cfg.get("lora_rank", 16)) != rig.RANK or int(cfg.get("lora_alpha", 32)) != int(rig.ALPHA):
            raise ValueError("the rig is r16 / alpha 32")

    def lora_layers(self) -> list:
        return [{k: d[k] for k in AD.KEYS} for d in self.lora]

    def parameters(self) -> list:
        return [d[k] for d in self.lora for k in AD.KEYS]

    def _routing(self, replay, T):
        dev = self.rig.DEV
        ids = torch.as_tensor(np.ascontiguousarray(replay.ids[:T]), dtype=torch.long, device=dev)
        return {v: ids[:, v, :].contiguous() for v in range(self.visits)}

    def _logprobs(self, row, replay, grad: bool, w4a4: bool):
        dev = self.rig.DEV
        ids = torch.as_tensor(row.ids, dtype=torch.long, device=dev)[None, :]
        T = ids.shape[1]
        routing = self._routing(replay, T) if replay is not None else None
        m = self.model
        if w4a4:
            self.rig.RIG["T"] = T
            m.reset()
            m.act_quant = True
            with torch.no_grad():
                res = m.forward(ids.reshape(-1), logits=None, routing_override=routing)
            h = res["hidden"]
        else:
            if grad:
                h, _ = m.hidden(ids, "train", routing=routing)
            else:
                with torch.no_grad():
                    h, _ = m.hidden(ids, "train", routing=routing)
        tgt = ids.reshape(-1)[1:]
        lp = m.target_logprobs(h[:-1], tgt, grad=grad)
        return torch.cat([torch.zeros(1, dtype=lp.dtype, device=lp.device), lp])

    def score(self, row, replay) -> np.ndarray:
        lp = self._logprobs(row, replay, grad=False, w4a4=(self.score_mode == "w4a4"))
        return lp.detach().double().cpu().numpy()

    def logprobs_grad(self, row, replay) -> torch.Tensor:
        return self._logprobs(row, replay, grad=True, w4a4=False)
