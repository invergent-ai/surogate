"""The engine-op trainer as a backend of the RL glue's kit shim (surogate/inkling_rl; contract in
its backend_fake.py / backend_rig.py): the forward IS the engine's (libsinfer.so through the trainer shim, bit for bit
the engine's prompt log-probs, M2), the backward is LEVEL 0 (ift_backward).

Contract:
  parameters() -> [layer][A13, B13, A2, B2] (rig layout; the shim's MasterAdamW writes bf16(master) into them);
  lora_layers() -> per physical layer {A13, B13, A2, B2};
  score(row, replay) -> np.float64 [T]: log p(ids[t] | ids[:t]) at step start, 0 at t = 0, no grad;
  logprobs_grad(row, replay) -> torch fp32 [T] with grad (same values: the same engine forward);
  replay_mixes_own = True: the forward routes itself and that IS the engine's routing; on covered positions the
  engine's captured ids are COMPARED (counted in self.r3_check), never injected;
  engine_sha = sha256 of the libsinfer.so loaded (the shim refuses a run whose rollouts came from another release).
Rule: rows are the engine's token ids; nothing here renders text.

cfg (the kit config's inkling_rl.engine): bin_dir (dir with libsinfer.so = the pinned release + SHA256SUMS),
shim (path of libsinfer_inkling_train.so; default <bin_dir>/serve_tests/...), artifact, device (0), lora_seed (0).
"""

from __future__ import annotations

import os

import numpy as np
import torch

from ift_engine import InklingEngine, sha256_file
import ift_backward as B
import ift_forward as F
import ift_long as LG


class EngineBackend:
    replay_mixes_own = True
    visits, top_k = 48, 8

    def __init__(self, cfg: dict):
        ec = ((cfg.get("inkling_rl") or {}).get("engine") or {})
        bin_dir = ec["bin_dir"]
        if ec.get("shim"):
            os.environ["IFT_SHIM_PATH"] = ec["shim"]
        if int(cfg.get("lora_rank", 16)) != B.RANK or int(cfg.get("lora_alpha", 32)) != int(B.ALPHA):
            raise ValueError("the engine backend trains expert LoRA r16 / alpha 32")
        self.engine_sha = sha256_file(os.path.join(bin_dir, "libsinfer.so"))
        self.eng = InklingEngine(bin_dir, ec["artifact"], int(ec.get("device", 0)))
        self.W = B.EngineWeights(self.eng)
        self.lora = B.make_lora(self.eng.dev, seed=int(ec.get("lora_seed", 0)), b_std=0.0)
        self.banks = B.EngineBanks(self.lora, self.eng.dev)
        self.tf = LG.TrainForwardLong(self.eng, self.W)  # the long-row path (== ift_backward's gradient on short rows)
        self.r3_check = {"rows": 0, "covered_positions": 0, "mismatched_positions": 0}

    # ------------------------------------------------------------------ contract
    def lora_layers(self) -> list:
        return [{k: d[k] for k in B.LORA_KEYS} for d in self.lora]

    def parameters(self) -> list:
        return [d[k] for d in self.lora for k in B.LORA_KEYS]

    def _ids(self, row) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(row.ids), dtype=torch.int32, device=self.eng.dev)

    def _check_routing(self, routing: torch.Tensor, replay) -> None:
        """routing int32 [48, T, 8] (this forward = the engine's) vs the rollout's capture [T, 48, 8] on covered rows."""
        if replay is None:
            return
        T = routing.shape[1]
        cov = torch.as_tensor(np.ascontiguousarray(replay.covered[:T]), device=routing.device)
        cap = torch.as_tensor(np.ascontiguousarray(replay.ids[:T]), dtype=torch.int32, device=routing.device)
        mine = routing.permute(1, 0, 2)
        bad = ((mine != cap).any(-1).any(-1) & cov).sum()
        self.r3_check["rows"] += 1
        self.r3_check["covered_positions"] += int(cov.sum())
        self.r3_check["mismatched_positions"] += int(bad)

    def score(self, row, replay) -> np.ndarray:
        self.banks.refresh(self.lora)
        ids = self._ids(row)
        with torch.no_grad():
            res = F.forward(self.eng, ids, save=False,
                            adapters_per_layer=[L["adapters"] for L in self.banks.layers])
            lp = F.logprobs(self.eng, res["hidden"], ids)
        self._check_routing(res["routing"], replay)
        out = np.zeros(ids.numel(), dtype=np.float64)
        out[1:] = lp.double().cpu().numpy()
        return out

    def logprobs_grad(self, row, replay) -> torch.Tensor:
        self.banks.refresh(self.lora)
        ids = self._ids(row)
        lp, res = self.tf.build(ids, self.lora, self.banks)
        return torch.cat([torch.zeros(1, dtype=lp.dtype, device=lp.device), lp])
