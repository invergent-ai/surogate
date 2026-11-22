"""Lane T8: one card's worker for the row-parallel step (t8_step.WorkerCore + the engine). The forward is the engine's; the backward is t8_lean.backward_row."""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import torch

import ift_backward as B
import ift_forward as F
import t8_center as RC
import t8_lean as TL
from ift_engine import InklingEngine
from t8_step import LORA_KEYS, ParamLayout, T8Config, T8GateError, WorkerCore, serve, verify_stack

GIB = 2 ** 30


def param_shapes() -> list:
    """Per physical layer {key: shape} of the expert LoRA (the layout of ift_backward.make_lora)."""
    return [{"A13": (B.E, B.RANK, B.D), "B13": (B.E, 2 * B.I, B.RANK), "A2": (B.E, B.RANK, B.I), "B2": (B.E, B.D, B.RANK)} for _ in range(24)]


def work_views(flat: torch.Tensor, shapes: list) -> list:
    """The shared work values (fp32 tensors holding bf16(master)) as per-layer dicts, in parameter order."""
    out, off = [], 0
    for sh in shapes:
        d = {}
        for k in LORA_KEYS:
            n = int(np.prod(sh[k]))
            d[k] = flat[off:off + n].view(sh[k])
            off += n
        out.append(d | {"scale": B.ALPHA / B.RANK})
    return out


class T8Worker(WorkerCore):
    def __init__(self, cfg: T8Config, work_flat: torch.Tensor, sum_flat: torch.Tensor, shapes: list, layout_layers: list, card: int = -1):
        self.stack_flags = verify_stack(cfg.stack)            # before anything runs: the path the gates proved
        self.card = card
        self.eng = InklingEngine(cfg.bin_dir, cfg.artifact, 0)
        self.dev = self.eng.dev
        self.W = B.EngineWeights(self.eng)
        self.v_first = len(self.eng.visits) - int(cfg.K)
        if self.v_first < 0:
            raise ValueError(f"K = {cfg.K}")
        self.lora = work_views(work_flat, shapes)
        self.banks = B.EngineBanks(self.lora, self.dev)
        self.buffers = None
        self.train_idx = tuple(j for j, k in enumerate(LORA_KEYS) if k in cfg.train_keys)
        self.setup_sums(ParamLayout(shapes, layout_layers, cfg.train_keys), sum_flat, cfg)

    def begin_step(self):
        super().begin_step()
        torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------------ adapter
    def refresh(self):
        self.banks.refresh(self.lora)
        torch.cuda.current_stream(self.eng.device).synchronize()

    def _adapters(self):
        return [L["adapters"] for L in self.banks.layers]

    def _ids(self, row):
        return torch.as_tensor(np.ascontiguousarray(row.ids), dtype=torch.int32, device=self.dev)

    # ------------------------------------------------------------------ gates and reads
    def check_zero_adapter(self, row) -> dict:
        """HARD GATE: every B == 0, and the forward with the adapter == the forward with no adapter (log-probs and routing), bitwise."""
        for lo in self.lora:
            if bool(lo["B13"].ne(0).any()) or bool(lo["B2"].ne(0).any()):
                raise T8GateError("zero-adapter check: the current masters have B != 0")
        ids = self._ids(row)
        with torch.no_grad():
            a = F.forward(self.eng, ids, save=False, adapters_per_layer=None)
            la = F.logprobs(self.eng, a["hidden"], ids)
            b = F.forward(self.eng, ids, save=False, adapters_per_layer=self._adapters())
            lb = F.logprobs(self.eng, b["hidden"], ids)
        bad_lp = int((la.view(torch.int32) != lb.view(torch.int32)).sum())
        bad_rt = int((a["routing"] != b["routing"]).sum())
        del a, b, la, lb, ids
        self.eng._ws = None
        torch.cuda.empty_cache()
        if bad_lp or bad_rt:
            raise T8GateError(f"zero-adapter check: T {row.T}: {bad_lp} log-probs and {bad_rt} routing ids differ from the base forward")
        return {"T": row.T, "log_probs_equal": True, "routing_equal": True}

    def logprobs_row(self, row) -> np.ndarray:
        """fp32 [T - 1], the workers' log-probs with the CURRENT banks (forward only; for the adapter round trip)."""
        ids = self._ids(row)
        with torch.no_grad():
            res = F.forward(self.eng, ids, save=False, adapters_per_layer=self._adapters())
            lp = F.logprobs(self.eng, res["hidden"], ids).cpu().numpy()
        del res, ids
        self.eng._ws = None
        torch.cuda.empty_cache()
        return lp

    # ------------------------------------------------------------------ WorkerCore hooks
    def _forward(self, row):
        T = row.T
        if self.buffers is None or T > self.buffers.T_max:
            if self.buffers is not None:
                self.buffers.close()
            self.buffers = TL.RowBuffers(self.eng.visits, self.v_first, T)
        ids = self._ids(row)
        res = TL.forward_keep(self.eng, ids, self._adapters(), self.buffers)
        return res["lp"].cpu().numpy(), {"ids": ids, "res": res, "qb": TL.qb_for(T)}

    def _check_routing(self, ctx, rp) -> int:
        routing = ctx["res"]["routing"]
        T = routing.shape[1]
        cov = torch.as_tensor(np.ascontiguousarray(rp.covered[:T]), device=self.dev)
        cap = torch.as_tensor(np.ascontiguousarray(rp.ids[:T]), dtype=torch.int32, device=self.dev)
        return int(((routing.permute(1, 0, 2) != cap).any(-1).any(-1) & cov).sum())

    def _support(self, ctx, row):
        ids = ctx["ids"]
        pos = torch.nonzero(torch.as_tensor(row.loss_mask[1:].copy(), device=self.dev))[:, 0]
        try:
            sup = RC.logits_and_support(SimpleNamespace(eng=self.eng), ctx["res"]["hidden"], pos, ids[1:][pos])
        except torch.OutOfMemoryError:
            raise
        except RuntimeError as e:                             # the support rebuild refused this row: a gate, not a crash
            raise T8GateError(f"SUPPORT GATE: T {row.T}: {e}") from e
        miss = int((~sup["inside"]).sum())
        if miss:
            raise T8GateError(f"SUPPORT GATE: T {row.T}: {miss} of {int(pos.numel())} sampled tokens are outside the rebuilt support")
        return sup

    def _backward_part(self, ctx, row, g_lp, sup, part):
        g = g_lp.to(self.dev)
        head = None if sup is None else (lambda W, hid, tg, gg: RC.head_grad_centered(W, hid, gg, sup))
        retries = 0
        while True:
            pg, ps = {}, {}

            def on_grad(v, layer, j, gj):
                if (layer, j) not in self.layout.index:
                    return
                c = gj.float().cpu()
                if j in self.seg_keys:
                    k = (part, self._segment_name(v), layer, LORA_KEYS[j])
                    ps[k] = c.clone() if k not in ps else ps[k].add_(c)
                pg[(layer, j)] = c if (layer, j) not in pg else pg[(layer, j)].add_(c)

            oom = False
            try:
                TL.backward_row(self.eng, self.W, ctx["ids"], ctx["res"], self.lora, B.ALPHA / B.RANK, self.v_first, g, self.buffers, on_grad,
                                head_grad=head, q_block=ctx["qb"], train_idx=self.train_idx)
            except torch.OutOfMemoryError:
                oom = True
            if not oom:
                return pg, ps, retries, ctx["qb"]
            # outside the except block: the traceback's frames no longer hold tensors
            pg, ps = None, None
            torch.cuda.empty_cache()
            if ctx["qb"] <= 8:
                raise RuntimeError(f"row of {row.T} tokens: out of card memory at query block 8")
            ctx["qb"] //= 2
            retries += 1

    def _release(self, ctx):
        # torch's pinned allocator rounds to powers of two and reuses a freed block only for its own size class (U0): a new class leaves the old
        # blocks held for good, so the host cache is emptied when the row's class changes
        T = int(ctx["ids"].numel())
        cls = (T * B.D * 2 - 1).bit_length()
        ctx.clear()
        torch.cuda.empty_cache()
        if getattr(self, "_pin_class", cls) != cls:
            torch._C._host_emptyCache()
        self._pin_class = cls

    def stats(self) -> dict:
        st = {"card": self.card, "stack": self.cfg.stack}
        for line in open("/proc/self/status"):
            k = line.split(":")[0]
            if k in ("VmHWM", "VmRSS", "RssAnon", "RssShmem"):
                st[k + "_gib"] = int(line.split()[1]) * 1024 / GIB
        st["card_peak_torch_gib"] = torch.cuda.max_memory_allocated() / GIB
        st["host_arena_gib"] = (self.buffers.bytes / GIB) if self.buffers is not None else 0.0
        return st

    def close(self):
        if self.buffers is not None:
            self.buffers.close()
            self.buffers = None
        self.eng.close()


def worker_main(conn, cfg, work_flat, sum_flat, shapes, layers, card=-1):
    """The target of a spawned worker process (the parent set CUDA_VISIBLE_DEVICES to this worker's card before starting it)."""
    try:
        w = T8Worker(cfg, work_flat, sum_flat, shapes, layers, card)
    except Exception:
        import traceback

        conn.send(("error", traceback.format_exc()[-4000:]))
        return
    conn.send(("ok", {"pid": os.getpid(), "card": card}))
    serve(conn, w)
