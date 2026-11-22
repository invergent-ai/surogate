"""Lane T8: the row-parallel GSPO step for W5 (lanes/T8/STEP-API.md; PLAN.md U1). One coordinator, one worker per card, each row on one card.

The per-row arithmetic is `inkling_rl.step.run_step`'s, split in two so that it can run on several cards:
 - plan_rows (coordinator): off-policy drop, refusals, R3 counters, the loss scale. Same order and rules as run_step, lines 126-167 and 193-196.
 - WorkerCore.process_row (worker): ONE forward per row (run_step's score pass and update pass run at the same weights, so their log-probs are the
   same bits), mismatch metric, TIS weights, seeds from the deployed loss, backward, row gradient added to the worker's sum.
Hard gates (STEP-API): replay (bitwise at every loss token of an on-policy row), routing where the rollout captured it, support (centred head),
finite gradients. A failed gate raises T8GateError in the coordinator and writes <run_dir>/STOP.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from surogate.inkling_rl.kitsrc import loss_api
from surogate.inkling_rl.records import Row, replay_rows
from surogate.inkling_rl.step import StepConfig, behavior_weights_sequence, behavior_weights_token, drop_off_policy, mismatch

LORA_KEYS = ("A13", "B13", "A2", "B2")


class T8GateError(RuntimeError):
    """A hard gate failed: the step returns no gradient."""


@dataclass
class T8Config:
    bin_dir: str = ""
    artifact: str = ""
    cards: tuple = (0,)
    K: int = 18                     # backward over the last K visits; 48 = full. The engine forward always runs all 48.
    head: str = "centred"           # "centred" | "plain"
    max_row_tokens: int = 58010
    segment_keys: tuple = ("B13", "B2")
    train_keys: tuple = ("A13", "B13", "A2", "B2")   # keys that get gradients; ("B13", "B2") skips the A vjps
    split_final: bool = False
    run_dir: str = ""               # the STOP file goes here
    lora_seed: int = 0
    visits: int = 48
    top_k: int = 8


@dataclass
class StepOut:
    grads: list
    segments: dict
    metrics: dict
    rows: list
    sense: str                      # "ascent" = gradient of sum weight x log pi; "descent" = gradient of the GSPO loss


# ---------------------------------------------------------------------------------------------------------------------
# coordinator side: which rows, and on which worker
# ---------------------------------------------------------------------------------------------------------------------
def plan_rows(rows: list, version, cfg: StepConfig, visits: int = 48, top_k: int = 8, replay_mixes_own: bool = True):
    """run_step's filters, in its order. Mutates the loss masks of refused rows, as run_step does. -> (live indices, metrics, r3, scale)."""
    m = {}
    dropped = drop_off_policy(rows, version, cfg.max_policy_lag)
    m["offpolicy_dropped_samples"] = float(dropped["samples"])
    m["offpolicy_dropped_tokens"] = float(dropped["tokens"])
    m["unversioned_samples"] = float(dropped["unversioned"])
    r3 = {"r3_rows": 0.0, "r3_replayed_tokens": 0.0, "r3_bad_rows": 0.0, "r3_shape_mismatches": 0.0, "r3_uncovered_interior": 0.0}
    refused_long = refused_r3 = refused_lag = 0
    if cfg.max_row_tokens:
        for r in rows:
            if r.loss_mask.any() and r.T > cfg.max_row_tokens:
                r.loss_mask = np.zeros_like(r.loss_mask)
                refused_long += 1
    live = [i for i, r in enumerate(rows) if r.loss_mask.any()]
    for i in list(live):
        rp, c = replay_rows(rows[i], visits, top_k)      # transient: the replay arrays of a whole batch would not fit the host
        if c["r3_uncovered_interior"] and not replay_mixes_own:
            rows[i].loss_mask = np.zeros_like(rows[i].loss_mask)
            refused_r3 += 1
            live.remove(i)
            continue
        if c["r3_uncovered_interior"] and version is not None and rows[i].versions is not None:
            vs = set(int(v) for v in rows[i].versions[rows[i].loss_mask])
            if vs and vs != {int(version)}:
                rows[i].loss_mask = np.zeros_like(rows[i].loss_mask)
                refused_lag += 1
                live.remove(i)
                continue
        del rp
        for k, v in c.items():
            r3[k] += v
    m["refused_long_rows"] = float(refused_long)
    m["r3_refused_rows"] = float(refused_r3)
    m["r3_lagged_uncovered_refused_rows"] = float(refused_lag)
    scale = float(max(sum(int(rows[i].loss_mask.sum()) for i in live), 1))
    if cfg.aggregation != "token":
        trials = {rows[i].trial_idx if rows[i].trial_idx is not None else ("row", i) for i in live}
        scale = float(max(len(trials), 1))
    return live, m, r3, scale


def assign_rows(lengths: dict, n_workers: int) -> list:
    """Longest row first, to the least-loaded worker (ties: lowest worker; equal lengths: lowest row index). A fixed function of the lengths.
    Cost of a row ~ its tokens. Each worker's list is longest first, so its host buffers are sized by its first row."""
    plan = [[] for _ in range(n_workers)]
    load = [0] * n_workers
    for i in sorted(lengths, key=lambda j: (-lengths[j], j)):
        w = min(range(n_workers), key=lambda k: (load[k], k))
        plan[w].append(i)
        load[w] += lengths[i]
    return plan


class ParamLayout:
    """A flat fp32 layout over the (layer, key) pairs that receive gradients. shapes: per layer {key: shape}."""

    def __init__(self, shapes: list, layers, keys):
        self.index = {}
        off = 0
        for L in layers:
            for j, k in enumerate(LORA_KEYS):
                if k in keys:
                    n = int(np.prod(shapes[L][k]))
                    self.index[(L, j)] = (off, n, tuple(shapes[L][k]))
                    off += n
        self.size = off

    def view(self, flat: torch.Tensor, L: int, j: int) -> torch.Tensor:
        o, n, s = self.index[(L, j)]
        return flat[o:o + n].view(s)


# ---------------------------------------------------------------------------------------------------------------------
# worker side: the arithmetic of one row
# ---------------------------------------------------------------------------------------------------------------------
class WorkerCore:
    """Subclasses give: visits_first (first visit of the backward), _forward, _check_routing, _support, _backward."""

    v_first = 0

    def setup_sums(self, layout: ParamLayout, flat: torch.Tensor, cfg: T8Config):
        self.layout, self.flat, self.cfg = layout, flat, cfg
        self.seg_keys = {LORA_KEYS.index(k) for k in cfg.segment_keys}
        self.seg_acc = {}

    def begin_step(self):
        self.flat.zero_()
        self.seg_acc = {}

    def _segment_name(self, v: int) -> str:
        import t8_lean

        return t8_lean.segment_of(v)

    def process_row(self, i: int, row: Row, job: dict) -> dict:
        t0 = time.time()
        cfg_step: StepConfig = job["step_cfg"]
        version, mode, scale = job["version"], job["seeds"], job["scale"]
        rp, _ = replay_rows(row, self.cfg.visits, self.cfg.top_k)
        lp32, ctx = self._forward(row)                       # fp32 [T - 1]: the engine's log-probs
        T = row.T
        mask = row.loss_mask
        lp = np.zeros(T, dtype=np.float64)
        lp[1:] = lp32.astype(np.float64)
        out = {"i": i, "T": T, "loss_tokens": int(mask.sum())}
        # REPLAY GATE: on-policy rows only (a lagged row was sampled by another adapter version)
        on_policy = True
        if version is not None and row.versions is not None:
            vs = set(int(v) for v in row.versions[mask])
            on_policy = vs <= {int(version), -1}
        if on_policy:
            want = row.engine_lp[1:].astype(np.float32)
            bad = int((lp32[mask[1:]] != want[mask[1:]]).sum())
            if bad:
                raise T8GateError(f"REPLAY GATE: row {i} (T {T}): {bad} of {int(mask.sum())} loss tokens differ from the rollout's log-probs")
        out["replay_checked"] = bool(on_policy)
        if rp is not None:
            miss = self._check_routing(ctx, rp)
            if miss:
                raise T8GateError(f"ROUTING GATE: row {i} (T {T}): {miss} covered positions differ from the rollout's capture")
        mm = mismatch(lp, row.engine_lp, mask)
        out["mm"] = mm
        w = None
        out["bw"] = None
        if cfg_step.behavior_is_cap is not None:
            fn = behavior_weights_token if cfg_step.behavior_is_level == "token" else behavior_weights_sequence
            w, x = fn(lp, row.engine_lp, mask, float(cfg_step.behavior_is_cap))
            if len(x):
                cap = float(cfg_step.behavior_is_cap)
                out["bw"] = (float(np.sum(x.astype(np.float64))), float(np.min(x)), float(np.max(x)), int(np.sum(x >= cap)), int(x.size))
        if mode == "gspo":
            _, grads_fn = loss_api()
            adv = np.full(T, row.advantage, dtype=np.float64)
            seeds, met = grads_fn(lp, lp, adv, mask, cfg_step.loss_config(), [(0, T)], teacher_logprobs=row.teacher_lp)
            if w is not None:
                seeds = seeds * w
            seeds_t = torch.as_tensor(seeds, dtype=torch.float32)
            # the coefficient autograd gives lp in run_step's loss = -(seeds * lp).sum() / scale
            g = (-(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(scale, dtype=torch.float32))) * seeds_t
            out["met"] = {k: float(met[k]) for k in ("policy_loss", "mismatch_kl", "is_masked")}
        elif mode == "weights":
            g = torch.as_tensor(float(row.notes["weight"]) * mask.astype(np.float32), dtype=torch.float32)
            out["met"] = {"policy_loss": 0.0, "mismatch_kl": 0.0, "is_masked": 0.0}
        else:
            raise ValueError(f"seeds {mode!r}")
        out["seq_log_ratio"] = 0.0                            # one forward: lp_new is lp_start by construction
        parts = [("main", mask)]
        if self.cfg.split_final:
            fin = np.asarray(row.notes["final_mask"], dtype=bool) & mask
            parts = [("main", mask & ~fin), ("final", fin)]
        sup = self._support(ctx, row) if self.cfg.head == "centred" else None
        out["decision_tokens"] = None if sup is None else int((sup["size"] >= 2).sum())
        row_grad, row_seg = {}, {}
        passes, retries, block = 0, 0, None
        for pname, pm in parts:
            if not pm.any():
                continue
            if sup is not None and not self._has_decision(sup, pm):
                continue                                      # every centred seed of this part is exactly 0, so is its gradient
            gp = g * torch.as_tensor(pm.astype(np.float32))
            part_grad, part_seg, r, block = self._backward_part(ctx, row, gp[1:], sup, pname)
            retries += r
            passes += 1
            for k, t in part_grad.items():
                row_grad[k] = t if k not in row_grad else row_grad[k].add_(t)
            for k, t in part_seg.items():
                row_seg[k] = t if k not in row_seg else row_seg[k].add_(t)
        for (L, j), t in row_grad.items():
            if not bool(torch.isfinite(t).all()):
                raise T8GateError(f"FINITE GATE: row {i} (T {T}): layer {L} {LORA_KEYS[j]} holds non-finite values")
            self.layout.view(self.flat, L, j).add_(t)
        for k, t in row_seg.items():
            self.seg_acc[k] = t if k not in self.seg_acc else self.seg_acc[k].add_(t)
        self._release(ctx)
        out.update({"passes": passes, "oom_retries": retries, "q_block": block, "seconds": time.time() - t0,
                    "grad_norm_by_segment": {f"{k[0]}.{k[1]}": v for k, v in self._seg_norms(row_seg).items()}})
        return out

    @staticmethod
    def _seg_norms(row_seg: dict) -> dict:
        sq = {}
        for (p, s, _L, _k), t in row_seg.items():
            sq[(p, s)] = sq.get((p, s), 0.0) + float(t.double().pow(2).sum())
        return {k: v ** 0.5 for k, v in sq.items()}

    @staticmethod
    def _has_decision(sup: dict, pm: np.ndarray) -> bool:
        pos = sup["row"].cpu().numpy()
        size = sup["size"].cpu().numpy()
        return bool(((size >= 2) & pm[1:][pos]).any())

    def _backward_part(self, ctx, row, g_lp, sup, part):
        raise NotImplementedError

    def _release(self, ctx):
        pass

    def finish_step(self) -> dict:
        return {"segments": self.seg_acc, "stats": self.stats()}

    def stats(self) -> dict:
        return {}


# ---------------------------------------------------------------------------------------------------------------------
# the coordinator
# ---------------------------------------------------------------------------------------------------------------------
class InlineHandle:
    """A worker object in this process (CPU tests)."""

    def __init__(self, worker):
        self.w = worker

    def call(self, name, *args):
        return getattr(self.w, name)(*args)

    def close(self):
        if hasattr(self.w, "close"):
            self.w.close()


class ProcHandle:
    """A worker in its own process, one request at a time over a pipe."""

    def __init__(self, proc, conn):
        self.proc, self.conn = proc, conn

    def call(self, name, *args):
        self.conn.send((name, args))
        kind, val = self.conn.recv()
        if kind == "gate":
            raise T8GateError(val)
        if kind == "error":
            raise RuntimeError(val)
        return val

    def close(self):
        try:
            self.conn.send(("__exit__", ()))
            self.proc.join(60)
        finally:
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(10)


def serve(conn, worker):
    """The worker process's loop: one request, one answer. A gate failure or any exception is an answer, never a silent exit."""
    import traceback

    while True:
        name, args = conn.recv()
        if name == "__exit__":
            if hasattr(worker, "close"):
                worker.close()
            conn.send(("ok", None))
            return
        try:
            conn.send(("ok", getattr(worker, name)(*args)))
        except T8GateError as e:
            conn.send(("gate", str(e)))
        except Exception:
            conn.send(("error", traceback.format_exc()[-4000:]))


def aggregate(results: list, m: dict, r3: dict, scale: float) -> dict:
    """run_step's metrics from the per-row results, in row order."""
    res = sorted(results, key=lambda r: r["i"])
    kl = [r["mm"][0] for r in res if r["mm"] is not None]
    lr = [r["mm"][1] for r in res if r["mm"] is not None]
    if kl:
        m["engine_mismatch_kl"] = float(np.mean(kl))
        m["engine_log_ratio"] = float(np.mean(lr))
    bw = [r["bw"] for r in res if r["bw"] is not None]
    if bw:
        n = sum(b[4] for b in bw)
        m.update({"engine_behavior_w_mean": sum(b[0] for b in bw) / n, "engine_behavior_w_min": min(b[1] for b in bw),
                  "engine_behavior_w_max": max(b[2] for b in bw), "engine_behavior_w_capped": sum(b[3] for b in bw) / n,
                  "engine_behavior_w_tokens": float(n)})
    tot = max(sum(r["loss_tokens"] for r in res), 1)
    for k in ("policy_loss", "mismatch_kl", "is_masked"):
        m[k] = sum(r["met"][k] * r["loss_tokens"] for r in res) / tot
    m["seq_log_ratio"] = float(np.mean([r["seq_log_ratio"] for r in res])) if res else 0.0
    m["total_tokens"] = sum(r["loss_tokens"] for r in res)
    m["expected_loss_scale"] = scale
    r3 = dict(r3)
    r3["r3_rows"] *= 2                                        # the kit counts R3 per pass over a row; two passes in run_step
    r3["r3_replayed_tokens"] *= 2
    m.update({k: float(v) for k, v in r3.items()})
    return m


def run_rows(handles: list, rows: list, live: list, job: dict) -> list:
    """Every worker runs its planned rows in order, one thread per worker. The first failure stops the others before their next row."""
    plan = assign_rows({i: rows[i].T for i in live}, len(handles))
    results, errors = [], []
    lock = threading.Lock()

    def one(w: int):
        try:
            for i in plan[w]:
                with lock:
                    if errors:
                        return
                r = handles[w].call("process_row", i, rows[i], job)
                r["worker"] = w
                with lock:
                    results.append(r)
        except Exception as e:
            with lock:
                errors.append(e)

    th = [threading.Thread(target=one, args=(w,), daemon=True) for w in range(len(handles))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if errors:
        gate = [e for e in errors if isinstance(e, T8GateError)]
        raise (gate[0] if gate else errors[0])
    return results


def sum_flats(flats: list) -> torch.Tensor:
    """The workers' fp32 sums added in worker order in fp64, rounded once to fp32."""
    total = flats[0].double().clone()
    for f in flats[1:]:
        total.add_(f.double())
    return total.float()


def write_stop(run_dir: str, why: str) -> None:
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "STOP"), "w") as f:
            json.dump({"why": why, "time": time.time()}, f)


def compute_step(handles: list, layout: ParamLayout, flats: list, shapes: list, cfg: T8Config, rows: list, version, step_cfg: StepConfig,
                 seeds: str = "gspo") -> StepOut:
    """One step over `rows` with the given workers (each already holds the current adapter)."""
    if seeds not in ("gspo", "weights"):
        raise ValueError(seeds)
    step_cfg = StepConfig(**{**asdict(step_cfg), "max_row_tokens": cfg.max_row_tokens})
    live, m, r3, scale = plan_rows(rows, version, step_cfg, cfg.visits, cfg.top_k)
    t0 = time.time()
    for h in handles:
        h.call("begin_step")
    job = {"step_cfg": step_cfg, "version": version, "seeds": seeds, "scale": scale}
    try:
        results = run_rows(handles, rows, live, job)
    except T8GateError as e:
        write_stop(cfg.run_dir, str(e))
        raise
    wall = time.time() - t0
    fin = [h.call("finish_step") for h in handles]
    total = sum_flats(flats)
    grads = []
    for L, sh in enumerate(shapes):
        for j, k in enumerate(LORA_KEYS):
            if (L, j) in layout.index:
                grads.append(layout.view(total, L, j).clone())
            else:
                grads.append(torch.zeros(tuple(sh[k]), dtype=torch.float32))
    segments = {}
    for f in fin:                                             # worker order
        for k in sorted(f["segments"], key=str):
            t = f["segments"][k]
            segments[k] = t.double() if k not in segments else segments[k].add_(t.double())
    segments = {k: t.float() for k, t in segments.items()}
    m = aggregate(results, m, r3, scale)
    tokens = sum(r["T"] for r in results)
    m["row_tokens"] = tokens
    m["wall_s"] = wall
    m["tok_per_s_per_card"] = tokens / max(wall, 1e-9) / len(handles)
    m["workers"] = [f["stats"] for f in fin]
    return StepOut(grads=grads, segments=segments, metrics=m, rows=sorted(results, key=lambda r: r["i"]),
                   sense="descent" if seeds == "gspo" else "ascent")


# ---------------------------------------------------------------------------------------------------------------------
# the trainer object agent-RX's loop holds (STEP-API.md)
# ---------------------------------------------------------------------------------------------------------------------
class T8Trainer:
    """One spawned worker per card. A script using it must guard its entry point with `if __name__ == "__main__"` (spawn re-imports it)."""

    def __init__(self, cfg: T8Config, masters=None):
        import torch.multiprocessing as mp

        import ift_backward as B
        import t8_lean as TL
        import t8_worker as TW
        from surogate.inkling_rl.shim import MasterAdamW

        self.cfg, self.B = cfg, B
        self.shapes = TW.param_shapes()
        n_all = sum(int(np.prod(s[k])) for s in self.shapes for k in LORA_KEYS)
        self.work_flat = torch.empty(n_all, dtype=torch.float32).share_memory_()
        self.work_layers = TW.work_views(self.work_flat, self.shapes)
        init = B.make_lora("cpu", seed=cfg.lora_seed, b_std=0.0)
        for d, lo in zip(self.work_layers, init):
            for k in LORA_KEYS:
                d[k].copy_(lo[k].detach())
        self.work = [torch.nn.Parameter(d[k], requires_grad=True) for d in self.work_layers for k in LORA_KEYS]
        self.opt = MasterAdamW(self.work, lr=5e-5, weight_decay=0.0, max_grad_norm=1.0)  # masters = the exact init; work = bf16(master)
        v_first = cfg.visits - int(cfg.K)
        if not 0 <= v_first < cfg.visits:
            raise ValueError(f"K = {cfg.K}")
        visits = [SimpleVisit(v) for v in range(cfg.visits)]
        self.layers = TL.trained_layers(visits, v_first)
        self.layout = ParamLayout(self.shapes, self.layers, cfg.train_keys)
        self.flats = [torch.zeros(self.layout.size, dtype=torch.float32).share_memory_() for _ in cfg.cards]
        ctx = mp.get_context("spawn")
        self.handles = []
        saved = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            for c, flat in zip(cfg.cards, self.flats):
                os.environ["CUDA_VISIBLE_DEVICES"] = str(c)
                here, there = ctx.Pipe()
                p = ctx.Process(target=TW.worker_main, args=(there, cfg, self.work_flat, flat, self.shapes, self.layers), daemon=True)
                p.start()
                there.close()
                self.handles.append(ProcHandle(p, here))
        finally:
            if saved is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved
        for h in self.handles:                                # each worker reports after its engine load
            kind, val = h.conn.recv()
            if kind != "ok":
                self.close()
                raise RuntimeError(f"worker failed to start: {val}")
        if masters is not None:
            self.set_masters(masters)

    # ------------------------------------------------------------------ adapter
    def _refresh(self):
        for h in self.handles:
            h.call("refresh")

    def masters(self) -> list:
        """The current fp32 masters in the rig layout (copies)."""
        out, i = [], 0
        for _ in self.shapes:
            out.append({k: self.opt.masters[i + j].detach().clone() for j, k in enumerate(LORA_KEYS)})
            i += len(LORA_KEYS)
        return out

    def set_masters(self, layers: list) -> None:
        i = 0
        with torch.no_grad():
            for d in layers:
                for k in LORA_KEYS:
                    self.opt.masters[i].copy_(torch.as_tensor(np.asarray(d[k]), dtype=torch.float32))
                    i += 1
        self.opt.sync_work()
        self._refresh()

    def export_adapter(self, out_dir: str, dtype: str = "BF16", meta: dict | None = None) -> str:
        from surogate.inkling_rl import adapter as AD

        layers = [{k: v.numpy() for k, v in d.items()} for d in self.masters()]
        return AD.export_adapter(layers, out_dir, self.B.RANK, int(self.B.ALPHA), dtype, metadata=meta)

    # ------------------------------------------------------------------ gates and reads
    def check_zero_adapter(self, rows: list) -> list:
        return [self.handles[0].call("check_zero_adapter", r) for r in rows]

    def logprobs(self, rows: list) -> list:
        """Per row fp32 [T - 1] with the CURRENT banks, forward only, rows spread over the workers by the step's plan."""
        plan = assign_rows({i: r.T for i, r in enumerate(rows)}, len(self.handles))
        out = [None] * len(rows)
        errs = []

        def one(w):
            try:
                for i in plan[w]:
                    out[i] = self.handles[w].call("logprobs_row", rows[i])
            except Exception as e:
                errs.append(e)

        th = [threading.Thread(target=one, args=(w,), daemon=True) for w in range(len(self.handles))]
        for t in th:
            t.start()
        for t in th:
            t.join()
        if errs:
            raise errs[0]
        return out

    # ------------------------------------------------------------------ the step
    def compute(self, rows: list, version, step_cfg: StepConfig, seeds: str = "gspo") -> StepOut:
        return compute_step(self.handles, self.layout, self.flats, self.shapes, self.cfg, rows, version, step_cfg, seeds)

    def apply(self, out: StepOut, lr: float) -> float:
        """The kit's fp32 AdamW on the masters. AdamW descends, so only a "descent" gradient is accepted (the sign cannot slip at the hand-over)."""
        if out.sense != "descent":
            raise ValueError('apply() takes a "descent" gradient (seeds="gspo"); for seeds="weights" make the update and call set_masters')
        for g in self.opt.opt.param_groups:
            g["lr"] = lr
        for p, g in zip(self.work, out.grads):
            p.grad = g
        norm = self.opt.step()
        self._refresh()
        return norm

    def close(self) -> None:
        for h in self.handles:
            try:
                h.close()
            except Exception:
                pass
        self.handles = []


class SimpleVisit:
    """Physical layer of a visit, without loading the engine (OPS.md section 1)."""

    def __init__(self, v: int):
        self.layer = v if v < 6 else (6 + (v - 6) % 12 if v < 42 else v - 24)
