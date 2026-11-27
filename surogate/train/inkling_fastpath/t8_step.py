"""The row-parallel GSPO step for Inkling-2-Mini (interface: STEP-API; plan item U1). One coordinator, one worker per card, each row on one card.

The per-row arithmetic is `inkling_rl.step.run_step`'s, split in two so that it can run on several cards:
 - plan_rows (coordinator): off-policy drop, refusals, R3 counters, the loss scale. Same order and rules as run_step, lines 126-167 and 193-196.
 - WorkerCore.process_row (worker): ONE forward per row (run_step's score pass and update pass run at the same weights, so their log-probs are the
   same bits), mismatch metric, TIS weights, seeds from the deployed loss, backward, row gradient added to the worker's sum.
Hard gates: replay (bitwise at every loss token of an on-policy row), routing where the rollout captured it, support (centred head),
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
# S3: S10 attention LoRA, per physical layer and slice (q, r, gate, sink, k, v of qkvr; o = wo_ud): A [R, k] and B [n, R]
ATTN_SLICES = ("q", "r", "g", "s", "k", "v", "o")
ATTN_KEYS = tuple(f"{s}{ab}" for s in ATTN_SLICES for ab in ("A", "B"))


def param_keys(attn_rank: int) -> tuple:
    """The per-layer key order of every flat layout, master list and gradient list (experts first, so j < 4 are the expert keys)."""
    return LORA_KEYS + (ATTN_KEYS if attn_rank else ())
# The code path is part of the gradient's definition: ift_long reads these at import, and its defaults are NOT the path the gates prove (review B1).
# A0 = the engine-op trainer's pre-lever stack in torch: the REFERENCE the kernel stack is checked against (tests/lever_check.py, tests/two_stack_grad.py).
# The trainer sets a stack's flags for its workers and every worker checks them after import.
STACKS = {"A0": {"IFT_MOE_GM": "0", "IFT_MOE_DET": "0", "IFT_FAST_GEMM": "0", "IFT_ATTN_ADAPT": "0", "IFT_CHUNK": "1024", "IFT_MOE_FAST": "0",
                 "IFT_ATTN_TC": "0", "IFT_TF32": "0", "IFT_MOE_ROWS": "8192", "IFT_MOE_PAD_ROWS": "0", "IFT_MANUAL": "1", "IFT_OFFLOAD_MIN": "8192",
                 "IFT_MOE_CUDA": "0", "IFT_ATTN_CUDA": "0", "IFT_STAGE_CUDA": "0"}}
# C1 = A0 with the backward on the CUDA kernel library (cuda_bwd; library path in IFT_BWD_LIB): the grouped expert backward, the relative
# attention and the two hand-written stage backwards. THE DEFAULT (DEFAULT_STACK) for supervised and RL steps; A0 is selected with
# T8Config.stack = "A0" or env T8_STACK=A0.
STACKS["C1"] = {**STACKS["A0"], "IFT_MOE_GM": "1", "IFT_MOE_CUDA": "1", "IFT_ATTN_CUDA": "1", "IFT_STAGE_CUDA": "1"}
DEFAULT_STACK = "C1"
WORKER_ENV = {"SUROGATE_INKLING_ATTN_EARLY_META": "0", "MALLOC_ARENA_MAX": "2", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
              "OMP_NUM_THREADS": "8", "PYTHONUNBUFFERED": "1"}
_FLAG_ATTR = {"IFT_MOE_GM": "MOE_GM", "IFT_MOE_DET": "MOE_DET", "IFT_FAST_GEMM": "FAST_GEMM", "IFT_ATTN_ADAPT": "ATTN_ADAPT", "IFT_CHUNK": "CHUNK",
              "IFT_MOE_FAST": "MOE_FAST", "IFT_ATTN_TC": "ATTN_TC", "IFT_TF32": "TF32", "IFT_MOE_ROWS": "MOE_ROWS", "IFT_MOE_PAD_ROWS": "MOE_PAD_ROWS",
              "IFT_MANUAL": "MANUAL", "IFT_OFFLOAD_MIN": "OFFLOAD_MIN", "IFT_MOE_CUDA": "MOE_CUDA", "IFT_ATTN_CUDA": "ATTN_CUDA",
              "IFT_STAGE_CUDA": "STAGE_CUDA"}


# A part of a row whose seeds are ALL exactly zero (e.g. a row with advantage 0 in "ppo" mode) has an exactly zero gradient: its backward is
# not run. The forward, the gates, the divisor and the sketch counters are unchanged, and so are the sums (adding zeros changes no bit).
# Not applied when segment sums are kept (there a zero row still leaves its zero entries). False = always run the backward.
SKIP_ZERO_SEED_PARTS = True
TIMING = os.environ.get("T8_TIMING", "0") == "1"      # per-row and per-step time split in the metrics (adds card synchronisations)


def _sync_now() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

# Bin edges of the per-row histogram of |log ratio| over the loss tokens ("ppo" mode): [0, 1e-6), then ten bins per decade up to 10^1.5, then
# everything above. A report only: nothing in the step reads it.
LOG_RATIO_EDGES = np.concatenate([[0.0], 10.0 ** (np.arange(-60, 16) / 10.0), [np.inf]])


class T8GateError(RuntimeError):
    """A hard gate failed: the step returns no gradient."""


def verify_stack(name: str) -> dict:
    """The flags ift_long was imported with == the named stack, else T8GateError. Returns them for the record."""
    import ift_long as LG

    got, bad = {}, []
    for env, want in STACKS[name].items():
        v = getattr(LG, _FLAG_ATTR[env])
        got[_FLAG_ATTR[env]] = v
        if (int(v) if isinstance(v, bool) else v) != int(want):
            bad.append(f"{_FLAG_ATTR[env]} = {v}, stack {name} needs {want}")
    if os.environ.get("SUROGATE_INKLING_ATTN_EARLY_META") != "0":
        bad.append("SUROGATE_INKLING_ATTN_EARLY_META is not 0")
    if bad:
        raise T8GateError("STACK GATE: " + "; ".join(bad))
    return got


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
    stack: str = os.environ.get("T8_STACK", DEFAULT_STACK)  # the code path (STACKS); set for the workers and checked in each of them
    attn_rank: int = 0              # S10 attention LoRA rank on the seven slices; 0 = none; 64 = the chosen rank (SkyRL: 128)
    attn_alpha: float = 32.0        # scale alpha / rank (0.5 at rank 64)
    allow_lagged: bool = False      # False: a row sampled by another adapter version stops the step (its replay gate could not run)
    row_timeout_s: float = 3600.0   # a worker that answers no request for this long ends the step
    device_sum: bool = os.environ.get("T8_DEVICE_SUM", "1" if os.environ.get("T8_STACK", DEFAULT_STACK) == "C1" else "0") == "1"   # (on with stack C1)
                                    # a row's gradient sums and the worker's step sum stay on the card and come to
                                    # the host once per step (the same fp32 additions in the same order, bitwise); rows above
                                    # device_sum_max_tokens, segment sums, sketches and lean_host keep the host path.
    device_sum_max_tokens: int = 16384
    lean_host: bool = False         # S4 (a): no per-row copy of a visited-once layer's gradient (added straight into the worker's sum; looped
                                    # layers keep their per-row sum until their last visit: same additions in the same order, bitwise).
                                    # A row that runs out of card memory after its first add is NOT retried: the step fails (STOP, masters restored).
    # per-attempt gradient sketches for the loop's signal read: for every attempt (row.trial_idx), the gradient of its summed loss-token log-prob term
    # BEFORE the advantage, on the attention keys of the named VISITS and on every sketch_expert_stride-th coordinate of each expert key (physical
    # layer totals), written to sketch_dir per worker at the end of the step. In "ppo" mode the row's gradient is divided by its seed factor -A / N
    # (exact at ratio 1, i.e. the first mini-batch; later: the sum of r_t x grad log pi over the unclipped tokens). Rows with A = 0 are counted, not sketched.
    sketch_visits: tuple = ()
    sketch_expert_stride: int = 0
    sketch_dir: str = ""
    empty_pinned_each_visit: bool = False  # S4 (b): torch's pinned pool emptied after every visit. G17 (g17-2): (a) + (b) together saved 1.1 GiB host
                                    # per worker at 57k but cost 15 % tok/s (171 -> 146); off by default
    # Constants of the backward (t8_lean.backward_row; 0 = the exact backward). A visit with an index below the value holds that path constant:
    # pattern_grad_from: its attention pattern (no gradient through the softmax scores into q, k and the relative bias; the value path stays);
    # gate_grad_from: the router's gate weights of its experts; bank_grad_from: a looped visit's two bank inputs (the previous pass's outputs).
    pattern_grad_from: int = 0
    gate_grad_from: int = 0
    bank_grad_from: int = 0


def validate_config(cfg: T8Config) -> None:
    """A misspelt option must not silently change the estimator (review M3)."""
    if cfg.head not in ("centred", "plain"):
        raise ValueError(f'head {cfg.head!r}: "centred" or "plain"')
    if cfg.attn_rank and not (1 <= int(cfg.attn_rank) <= 64):
        raise ValueError(f"attn_rank {cfg.attn_rank}: the engine's S10 kernels take ranks 1..64")
    if cfg.attn_rank and cfg.attn_alpha / cfg.attn_rank not in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0):
        raise ValueError("attn_alpha / attn_rank must be a power of two (the store's bf16 fold is then exact)")
    if (cfg.sketch_visits or cfg.sketch_expert_stride) and not cfg.sketch_dir:
        raise ValueError("sketches need sketch_dir")
    if cfg.sketch_visits and not cfg.attn_rank:
        raise ValueError("sketch_visits name attention keys: attn_rank must be > 0")
    if any(not 0 <= int(v) < int(cfg.visits) for v in cfg.sketch_visits):
        raise ValueError(f"sketch_visits {cfg.sketch_visits}")
    for name in ("pattern_grad_from", "gate_grad_from", "bank_grad_from"):
        v = getattr(cfg, name)
        if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= int(cfg.visits):
            raise ValueError(f"{name} {v!r}: a visit index 0..{cfg.visits}")
    if cfg.bank_grad_from and cfg.lean_host:
        raise ValueError("bank_grad_from skips visits of the looped layers: lean_host adds a layer's sum at its last visit and would not fire")
    if cfg.lean_host and cfg.split_final:
        raise ValueError("lean_host adds straight into the step's sum: with split_final the two parts would be added in another order")
    if cfg.attn_rank and not set(cfg.train_keys) & set(ATTN_KEYS):
        raise ValueError("attn_rank > 0 but no attention key in train_keys: the attention adapter would run at B = 0 for the whole run")
    if not cfg.train_keys or set(cfg.train_keys) - set(param_keys(cfg.attn_rank)):
        raise ValueError(f"train_keys {cfg.train_keys!r}")
    if set(cfg.segment_keys) - set(cfg.train_keys):
        raise ValueError(f"segment_keys {cfg.segment_keys!r} must be among train_keys {cfg.train_keys!r}")
    if not 1 <= int(cfg.K) <= int(cfg.visits):
        raise ValueError(f"K = {cfg.K}")
    if cfg.stack not in STACKS:
        raise ValueError(f"stack {cfg.stack!r}")
    if cfg.head == "centred" and cfg.allow_lagged:
        raise ValueError("the centred seed needs the sampling policy's own support: allow_lagged is refused with head='centred'")
    if len(set(cfg.cards)) != len(cfg.cards) or not cfg.cards:
        raise ValueError(f"cards {cfg.cards!r}")


def lagged(row: Row, version) -> bool:
    """True when the row's loss tokens were sampled by an adapter version other than `version` (unversioned tokens carry -1)."""
    if version is None or row.versions is None:
        return False
    return bool(set(int(v) for v in row.versions[row.loss_mask]) - {int(version), -1})


def check_rows(rows: list, live: list, version, cfg: T8Config) -> None:
    if not live:
        raise T8GateError("EMPTY STEP: no row has loss tokens after the refusals")
    for i in live:
        if bool(rows[i].loss_mask[0]):
            raise T8GateError(f"row {i}: loss mask set on position 0 (no prefix predicts it)")
    late = [i for i in live if lagged(rows[i], version)]
    if late and not cfg.allow_lagged:
        raise T8GateError(f"VERSION GATE: {len(late)} rows (first: {late[0]}) were not sampled by adapter version {version}: their replay gate cannot run")


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
        self.keys = tuple(shapes[0])
        off = 0
        for L in layers:
            for j, k in enumerate(self.keys):
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
def accumulate_row(w, run, T: int, part: str, qb: int, visit_layers: list, empty_cache):
    sk_v = getattr(w, "sk_visits", set())
    sk_s = getattr(w, "sk_stride", 0)
    if sk_v or sk_s:
        if getattr(w, "_row_sketch", None) is None:
            w._row_sketch = {}
        row_sk = w._row_sketch                               # several parts of one row add into the same sketch
    else:
        row_sk = None
    """One row's backward and where its gradient goes (worker w: layout, flat, seg_keys, cfg, v_first, _segment_name).
    run(on_grad, q_block) runs the backward; it may raise torch.OutOfMemoryError.
    Default: the row's sums are collected per (layer, key) and returned (the caller checks them and adds them); an out-of-memory row is retried
    with the query block halved (floor 8).
    lean_host: a tensor's row sum goes into w.flat at its layer's LAST visit (the same fp32 additions in the same order as the default); a row
    that runs out of memory AFTER its first such addition raises (no retry: w.flat now holds part of the row; the step must fail, never go on)."""
    lean = bool(w.cfg.lean_host)
    dev = (bool(getattr(w.cfg, "device_sum", False)) and not lean and row_sk is None and not w.seg_keys
           and T <= int(getattr(w.cfg, "device_sum_max_tokens", 0)))
    last = {}
    for v in range(w.v_first, len(visit_layers)):
        L = visit_layers[v]
        last[L] = min(last.get(L, v), v)
    retries = 0
    while True:
        pg, ps = {}, {}
        added = [False]
        sk_try = {} if row_sk is not None else None          # this attempt's contributions; merged only if the backward completes

        def on_grad(v, layer, j, gj):
            if (layer, j) not in w.layout.index:
                return
            if dev:                                          # the row's sums stay on the card: the same fp32 additions in the same order
                if (layer, j) not in pg:
                    pg[(layer, j)] = gj.detach().to(torch.float32, copy=True)
                else:
                    pg[(layer, j)].add_(gj.detach().to(torch.float32))
                return
            if lean and not bool(torch.isfinite(gj).all()):
                raise T8GateError(f"FINITE GATE: T {T}: layer {layer} {w.layout.keys[j]} (visit {v}) holds non-finite values")
            c = gj.detach().to("cpu", torch.float32, copy=True)  # always a new tensor (a CPU gj would otherwise be added into in place)
            if sk_try is not None:
                key = w.layout.keys[j]
                if j >= 4 and v in sk_v:                     # an attention key of a named visit (layer run)
                    kk = ("visit", v, key)
                    sk_try[kk] = c.clone() if kk not in sk_try else sk_try[kk].add_(c)
                if j < 4 and sk_s:                           # every sk_s-th coordinate of the expert key, physical-layer total
                    kk = ("layer", layer, key)
                    smp = c.reshape(-1)[::sk_s].clone()
                    sk_try[kk] = smp if kk not in sk_try else sk_try[kk].add_(smp)
            if j in w.seg_keys:
                k = (part, w._segment_name(v), layer, w.layout.keys[j])
                ps[k] = c.clone() if k not in ps else ps[k].add_(c)
            pg[(layer, j)] = c if (layer, j) not in pg else pg[(layer, j)].add_(c)
            if lean and v == last[layer]:
                w.layout.view(w.flat, layer, j).add_(pg.pop((layer, j)))
                added[0] = True

        oom = False
        cb = on_grad
        if TIMING:
            def cb(v, layer, j, gj):                          # the hand-over's own time: the card's queued work is finished first
                t = _sync_now()
                on_grad(v, layer, j, gj)
                w._t_hand = getattr(w, "_t_hand", 0.0) + time.time() - t
        try:
            run(cb, qb)
        except torch.OutOfMemoryError:
            oom = True
        if not oom:
            if sk_try is not None:
                for k, x in sk_try.items():
                    row_sk[k] = x if k not in row_sk else row_sk[k].add_(x)
            return pg, ps, retries, qb
        pg, ps = None, None                                   # outside the except block: the traceback's frames no longer hold tensors
        empty_cache()
        if added[0]:
            raise RuntimeError(f"row of {T} tokens: out of card memory at query block {qb} after part of its gradient was added to the step's sum "
                               "(lean_host): the step fails; it is not retried and returns no gradient")
        if qb <= 8:
            raise RuntimeError(f"row of {T} tokens: out of card memory at query block 8")
        qb //= 2
        retries += 1


class WorkerCore:
    """Subclasses give: visits_first (first visit of the backward), _forward, _check_routing, _support, _backward."""

    v_first = 0

    def setup_sums(self, layout: ParamLayout, flat: torch.Tensor, cfg: T8Config):
        self.layout, self.flat, self.cfg = layout, flat, cfg
        self.seg_keys = {layout.keys.index(k) for k in cfg.segment_keys}
        self.seg_acc = {}
        self.sketch, self.sketch_meta = {}, {}
        self.sk_visits = set(int(v) for v in getattr(cfg, "sketch_visits", ()))
        self.sk_stride = int(getattr(cfg, "sketch_expert_stride", 0))

    def begin_step(self):
        self.flat.zero_()
        self._dev_flat, self._flat_dirty = None, False
        self.seg_acc = {}
        self.sketch, self.sketch_meta = {}, {}

    def _segment_name(self, v: int) -> str:
        import t8_lean

        return t8_lean.segment_of(v)

    def process_row(self, i: int, row: Row, job: dict) -> dict:
        t0 = time.time()
        cfg_step: StepConfig = job["step_cfg"]
        version, mode, scale = job["version"], job["seeds"], job["scale"]
        rp, _ = replay_rows(row, self.cfg.visits, self.cfg.top_k)
        lp32, ctx = self._forward(row)                       # fp32 [T - 1]: the engine's log-probs
        if TIMING:
            t_fwd = _sync_now()
            self._t_hand = 0.0
        T = row.T
        mask = row.loss_mask
        lp = np.zeros(T, dtype=np.float64)
        lp[1:] = lp32.astype(np.float64)
        out = {"i": i, "T": T, "loss_tokens": int(mask.sum())}
        # REPLAY GATE: on-policy rows only (a lagged row was sampled by another adapter version). In "ppo" mode it is read on the OLD log-probs (the
        # trainer's forward at the batch's start weights, recomputed as SkyRL does): after a mini-batch's optimizer step the current forward differs by design.
        on_policy = not lagged(row, version)
        old32 = None
        if mode == "ppo":
            old32 = np.asarray(row.notes["old_lp"], dtype=np.float32)
            if old32.shape != lp32.shape:
                raise T8GateError(f"row {i}: old_lp has {old32.shape[0]} entries for {lp32.shape[0]} targets")
            if job.get("minibatch_index", 0) == 0:
                bad0 = int((lp32[mask[1:]].view(np.int32) != old32[mask[1:]].view(np.int32)).sum())
                if bad0:  # the first mini-batch runs at the start weights: its forward must be the old log-probs, bit for bit
                    raise T8GateError(f"STALE GATE: row {i} (T {T}): {bad0} loss tokens differ from the recomputed old log-probs at unchanged weights")
                out["stale_checked"] = True
        if on_policy:
            want = row.engine_lp[1:].astype(np.float32)
            have = old32 if old32 is not None else lp32
            bad = int((have[mask[1:]] != want[mask[1:]]).sum())
            if bad:
                raise T8GateError(f"REPLAY GATE: row {i} (T {T}): {bad} of {int(mask.sum())} loss tokens differ from the rollout's log-probs")
        out["replay_checked"] = bool(on_policy)
        fm = row.notes.get("final_mask")
        if on_policy and fm is not None:  # the final-reply tokens the replay gate just covered (they are in the loss)
            out["replay_final_tokens"] = int((np.asarray(fm, dtype=bool) & mask).sum())
        # the routing gate compares THIS forward's own routing with the rollout's capture: in "ppo" mode only the first mini-batch runs at the
        # start weights (proven by the stale gate); later mini-batches run at moved weights, where a near-tie routing decision may flip (review B1)
        if rp is not None and on_policy and (mode != "ppo" or job.get("minibatch_index", 0) == 0):
            miss = self._check_routing(ctx, rp)
            if miss:
                raise T8GateError(f"ROUTING GATE: row {i} (T {T}): {miss} covered positions differ from the rollout's capture")
        if old32 is not None:
            old = np.zeros(T, dtype=np.float64)
            old[1:] = old32.astype(np.float64)
            mm = mismatch(old, row.engine_lp, mask)              # trainer (old) vs rollout: what SkyRL logs as rollout_train_logprobs_abs_diff
            dd = np.abs(old[mask] - row.engine_lp[mask]) if mask.any() else np.zeros(0)
            out["rollout_train_abs_diff"] = (float(dd.sum()), float(dd.max()) if dd.size else 0.0)
        else:
            mm = mismatch(lp, row.engine_lp, mask)
        out["mm"] = mm
        w = None
        out["bw"] = None
        if cfg_step.behavior_is_cap is not None and mode != "ppo":  # SkyRL's preset applies no off-policy correction
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
        elif mode == "ppo":
            import t8_skyrl as KS

            adv = np.full(T, row.advantage, dtype=np.float64)
            seeds, rloss, met = KS.dual_clip_seeds(lp, old, adv, mask, int(scale), job.get("skyrl", KS.SkyRLLoss()))
            g = torch.as_tensor(seeds, dtype=torch.float32)   # d loss / d logp, the 1 / mini-batch-tokens already inside (token mean)
            n_ = max(int(mask.sum()), 1)
            out["met"] = {"policy_loss": rloss * scale / n_, "mismatch_kl": 0.0, "is_masked": met["clipped_tokens"] / n_}
            out["ppo"] = met
            out["seq_log_ratio_ppo"] = float(np.mean(lp[mask] - old[mask])) if mask.any() else 0.0
            lr_abs = np.abs(lp[mask] - old[mask]) if mask.any() else np.zeros(0)
            out["log_ratio_abs"] = (float(lr_abs.sum()), float(lr_abs.max()) if lr_abs.size else 0.0)   # sum and max of |log ratio| over the loss tokens
            out["log_ratio_hist"] = np.histogram(lr_abs, bins=LOG_RATIO_EDGES)[0].astype(np.int64).tolist()   # counts per LOG_RATIO_EDGES bin
        elif mode == "weights":
            g = torch.as_tensor(float(row.notes["weight"]) * mask.astype(np.float32), dtype=torch.float32)
            out["met"] = {"policy_loss": 0.0, "mismatch_kl": 0.0, "is_masked": 0.0}
        else:
            raise ValueError(f"seeds {mode!r}")
        out["seq_log_ratio"] = out.pop("seq_log_ratio_ppo", 0.0)  # gspo / weights: one forward, lp_new is lp_start by construction
        parts = [("main", mask)]
        if self.cfg.split_final:
            fin = np.asarray(row.notes["final_mask"], dtype=bool) & mask
            parts = [("main", mask & ~fin), ("final", fin)]
        sup = self._support(ctx, row) if self.cfg.head == "centred" else None
        out["decision_tokens"] = None if sup is None else int((sup["size"] >= 2).sum())
        row_grad, row_seg = {}, {}
        passes, retries, block, zero_parts = 0, 0, None, 0
        for pname, pm in parts:
            if not pm.any():
                continue
            if sup is not None and not self._has_decision(sup, pm):
                continue                                      # every centred seed of this part is exactly 0, so is its gradient
            gp = g * torch.as_tensor(pm.astype(np.float32))
            if SKIP_ZERO_SEED_PARTS and not self.seg_keys and not bool(gp.ne(0).any()):
                zero_parts += 1                               # every seed exactly 0: the gradient is exactly 0, no backward
                continue
            part_grad, part_seg, r, block = self._backward_part(ctx, row, gp[1:], sup, pname)
            retries += r
            passes += 1
            for k, t in part_grad.items():
                row_grad[k] = t if k not in row_grad else row_grad[k].add_(t)
            for k, t in part_seg.items():
                row_seg[k] = t if k not in row_seg else row_seg[k].add_(t)
        rs = getattr(self, "_row_sketch", None)
        if rs is None and zero_parts and (self.sk_visits or self.sk_stride):
            rs = {}                                           # no backward ran: the row is still counted in its attempt's sketch record
        if rs is not None:
            self._sketch_row(row, rs, mode, scale)
            self._row_sketch = None
        out["zero_seed_parts"] = zero_parts
        if TIMING:
            t_bwd = _sync_now()
        tgt = self._sum_target(next(iter(row_grad.values())).device) if row_grad else self.flat
        for (L, j), t in row_grad.items():
            if not bool(torch.isfinite(t).all()):
                raise T8GateError(f"FINITE GATE: row {i} (T {T}): layer {L} {self.layout.keys[j]} holds non-finite values")
            self.layout.view(tgt, L, j).add_(t if t.device == tgt.device else t.to(tgt.device))
        for k, t in row_seg.items():
            self.seg_acc[k] = t if k not in self.seg_acc else self.seg_acc[k].add_(t)
        self._release(ctx)
        if TIMING:
            t_end = _sync_now()
            out["timing"] = {"forward": t_fwd - t0, "backward": t_bwd - t_fwd - self._t_hand, "hand_over": self._t_hand, "row_sum": t_end - t_bwd}
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

    def _sketch_row(self, row, rs: dict, mode: str, scale: float) -> None:
        if mode == "ppo":
            factor = -float(row.advantage) / float(scale)
        elif mode == "weights":
            factor = float(row.notes["weight"])
        else:
            raise ValueError(f"sketches are defined for seeds 'ppo' and 'weights', not {mode!r}")
        t = row.trial_idx
        meta = self.sketch_meta.setdefault(t, {"rows": 0, "rows_skipped_zero_factor": 0, "advantage": float(row.advantage), "group": row.group_idx})
        if factor == 0.0:
            meta["rows_skipped_zero_factor"] += 1
            return
        meta["rows"] += 1
        d = self.sketch.setdefault(t, {})
        for k, x in rs.items():
            y = x / factor
            d[k] = y if k not in d else d[k].add_(y)

    def _sum_target(self, device) -> torch.Tensor:
        """Where a row's sums are added: the worker's host sum, or (device_sum, a row whose sums are on the card) its copy on the card. One
        running sum at any time: the copy starts from the host sum and is written back before a host-side row and at the step's end, so the
        fp32 additions and their order are those of the host path."""
        if device.type == "cuda":
            if getattr(self, "_dev_flat", None) is None:
                try:
                    self._dev_flat = self.flat.to(device) if self._flat_dirty else torch.zeros(self.flat.shape, dtype=torch.float32, device=device)
                except torch.OutOfMemoryError:
                    self._dev_flat = None
            if self._dev_flat is not None:
                return self._dev_flat
        self._flush_device_sum()
        self._flat_dirty = True
        return self.flat

    def _flush_device_sum(self) -> None:
        if getattr(self, "_dev_flat", None) is not None:
            self.flat.copy_(self._dev_flat)
            self._dev_flat, self._flat_dirty = None, True

    def finish_step(self) -> dict:
        self._flush_device_sum()
        written = []
        sdir = getattr(self.cfg, "sketch_dir", "")
        if sdir and self.sketch_meta:
            os.makedirs(sdir, exist_ok=True)
            tag = getattr(self, "card", os.getpid())
            for t, meta in self.sketch_meta.items():
                path = os.path.join(sdir, f"sketch_trial{t}_w{tag}.pt")
                torch.save({"trial": t, **meta, "tensors": self.sketch.get(t, {}), "visits": sorted(self.sk_visits),
                            "expert_stride": self.sk_stride}, path + ".tmp")
                os.replace(path + ".tmp", path)
                written.append(path)
        self.sketch, self.sketch_meta = {}, {}
        out = self._finish_segments()
        out["sketch_files"] = written
        return out

    def _finish_segments(self) -> dict:
        """The segment sums as ONE tensor + an index (one file descriptor per worker over the pipe, whatever K and the keys)."""
        keys = sorted(self.seg_acc, key=str)
        index = [(k, tuple(self.seg_acc[k].shape)) for k in keys]
        flat = torch.cat([self.seg_acc[k].reshape(-1) for k in keys]) if keys else torch.zeros(0, dtype=torch.float32)
        return {"seg_index": index, "seg_flat": flat, "stats": self.stats()}

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

    def __init__(self, proc, conn, timeout_s: float = 3600.0):
        self.proc, self.conn, self.timeout_s, self.broken = proc, conn, timeout_s, False

    def call(self, name, *args):
        if self.broken:
            raise RuntimeError("this worker missed a deadline earlier; its answers can no longer be matched to requests")
        self.conn.send((name, args))
        if not self.conn.poll(self.timeout_s):
            self.broken = True
            raise TimeoutError(f"worker pid {self.proc.pid}: no answer to {name} in {self.timeout_s:.0f} s")
        kind, val = self.conn.recv()
        if kind == "gate":
            raise T8GateError(val)
        if kind == "error":
            raise RuntimeError(val)
        return val

    def close(self):
        try:
            if not self.broken:
                self.conn.send(("__exit__", ()))
                self.proc.join(60)
        except (BrokenPipeError, ConnectionResetError, EOFError, OSError):
            pass
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
    n = flats[0].numel()
    total = torch.empty(n, dtype=torch.float32)
    step = 1 << 20                                            # in cache-sized pieces: the same fp64 additions per element, in worker order
    for o in range(0, n, step):
        acc = flats[0][o:o + step].double()
        for f in flats[1:]:
            acc.add_(f[o:o + step])                           # fp32 -> fp64 is exact
        total[o:o + step] = acc.float()
    return total


def write_stop(run_dir: str, why: str) -> None:
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "STOP"), "w") as f:
            json.dump({"why": why, "time": time.time()}, f)


def compute_step(handles: list, layout: ParamLayout, flats: list, shapes: list, cfg: T8Config, rows: list, version, step_cfg: StepConfig,
                 seeds: str = "gspo", minibatch_index: int = 0) -> StepOut:
    """One step over `rows` with the given workers (each already holds the current adapter)."""
    if seeds not in ("gspo", "weights", "ppo"):
        raise ValueError(seeds)
    validate_config(cfg)
    step_cfg = StepConfig(**{**asdict(step_cfg), "max_row_tokens": cfg.max_row_tokens})
    live, m, r3, scale = plan_rows(rows, version, step_cfg, cfg.visits, cfg.top_k)
    t0 = time.time()
    job = {"step_cfg": step_cfg, "version": version, "seeds": seeds, "scale": scale, "minibatch_index": int(minibatch_index)}
    if seeds == "ppo" and step_cfg.aggregation != "token":
        raise ValueError("SkyRL's token mean needs aggregation 'token' (scale = the mini-batch's loss tokens)")
    if seeds == "ppo" and cfg.head != "plain":
        raise ValueError("seeds='ppo' is SkyRL's estimator on log pi: head must be 'plain' (the preset samples top-p 1.0 / top-k off, no score centering)")
    try:
        check_rows(rows, live, version, cfg)
        for h in handles:
            h.call("begin_step")
        t_begin = time.time()
        results = run_rows(handles, rows, live, job)
    except Exception as e:                                    # a gate, or a worker failure (e.g. lean_host's no-retry path): STOP, no gradient
        write_stop(cfg.run_dir, f"{type(e).__name__}: {e}"[:2000])
        raise
    wall = time.time() - t0
    fin = [h.call("finish_step") for h in handles]
    t_fin = time.time()
    total = sum_flats(flats)
    t_sum = time.time()
    grads = []
    for L, sh in enumerate(shapes):
        for j, k in enumerate(layout.keys):
            if (L, j) in layout.index:
                grads.append(layout.view(total, L, j).clone())
            else:
                grads.append(torch.zeros(tuple(sh[k]), dtype=torch.float32))
    segments = {}
    for f in fin:                                             # worker order
        off = 0
        for k, shape in f["seg_index"]:
            n = int(np.prod(shape))
            t = f["seg_flat"][off:off + n].view(shape)
            off += n
            segments[k] = t.double() if k not in segments else segments[k].add_(t.double())
    segments = {k: t.float() for k, t in segments.items()}
    m = aggregate(results, m, r3, scale)
    m["replay_checked_rows"] = sum(1 for r in results if r["replay_checked"])
    m["replay_skipped_rows"] = sum(1 for r in results if not r["replay_checked"])
    if m["replay_skipped_rows"] and not cfg.allow_lagged:
        write_stop(cfg.run_dir, "replay gate skipped on a row")
        raise T8GateError(f"VERSION GATE: the replay gate was skipped on {m['replay_skipped_rows']} rows")
    m["stack"] = cfg.stack
    m["stale_checked_rows"] = sum(1 for r in results if r.get("stale_checked"))
    m["zero_seed_rows"] = sum(1 for r in results if r.get("zero_seed_parts") and not r["passes"])   # rows whose backward was not run (all seeds 0)
    m["replay_checked_final_tokens"] = sum(r.get("replay_final_tokens", 0) for r in results)
    dd = [r["rollout_train_abs_diff"] for r in results if "rollout_train_abs_diff" in r]
    if dd:  # SkyRL's logged statistic, reported beside the bitwise replay gate
        m["rollout_train_logprobs_abs_diff_mean"] = sum(x[0] for x in dd) / max(m["total_tokens"], 1)
        m["rollout_train_logprobs_abs_diff_max"] = max(x[1] for x in dd)
    pp = [r["ppo"] for r in results if "ppo" in r]
    if pp:
        m["ppo_clip_ratio"] = sum(x["clipped_tokens"] for x in pp) / max(sum(x["loss_tokens"] for x in pp), 1.0)
        m["ppo_ratio_max"] = max(x["ratio_max"] for x in pp)
    tokens = sum(r["T"] for r in results)
    m["row_tokens"] = tokens
    m["wall_s"] = wall
    m["tok_per_s_per_card"] = tokens / max(wall, 1e-9) / len(handles)
    m["workers"] = [f["stats"] for f in fin]
    if TIMING:
        tm = {"begin_step": t_begin - t0, "rows": wall - (t_begin - t0), "finish_step": t_fin - t0 - wall, "sum_flats": t_sum - t_fin, "after_sum": time.time() - t_sum}
        per = [r["timing"] for r in results if "timing" in r]
        tm["row_parts_worker_seconds"] = {k: sum(x[k] for x in per) for k in ("forward", "backward", "hand_over", "row_sum")}
        tm["row_seconds_sum"] = sum(r["seconds"] for r in results)
        m["timing"] = tm
        print("T8_TIMING compute " + json.dumps({k: (round(v, 3) if isinstance(v, float) else {a: round(b, 2) for a, b in v.items()}) for k, v in tm.items()}), flush=True)
    m["sketch_files"] = [p for f in fin for p in f.get("sketch_files", [])]
    return StepOut(grads=grads, segments=segments, metrics=m, rows=sorted(results, key=lambda r: r["i"]),
                   sense="ascent" if seeds == "weights" else "descent")


# ---------------------------------------------------------------------------------------------------------------------
def fast_sync(base):
    """`base` (the kit's MasterAdamW) with the work values' bf16 rounding done by torch's float32 -> bfloat16 conversion instead of the kit's numpy
    routine: 0.5 s instead of 2.7 s per optimizer step on the expert adapter. The two give the same bits for every finite float32 (compared over
    all 2^32 - 2^24 finite bit patterns); a non-finite master is refused, as the kit's routine does."""

    class FastSync(base):
        def sync_work(self) -> None:
            with torch.no_grad():
                for p, m in zip(self.work, self.masters):
                    m = m.detach()
                    if not bool(torch.isfinite(m).all()):
                        raise ValueError("adapter values must be finite")
                    p.copy_(m.to(torch.bfloat16).to(p.device, p.dtype))

    return FastSync


# the trainer object the RL loop holds
# ---------------------------------------------------------------------------------------------------------------------
class T8Trainer:
    """One spawned worker per card. A script using it must guard its entry point with `if __name__ == "__main__"` (spawn re-imports it)."""

    def __init__(self, cfg: T8Config, masters=None):
        import torch.multiprocessing as mp

        import ift_backward as B
        import t8_lean as TL
        import t8_worker as TW
        from surogate.inkling_rl.shim import MasterAdamW

        validate_config(cfg)
        physical = self._physical_cards(cfg.cards)
        self.cfg, self.B = cfg, B
        self.shapes = TW.param_shapes(cfg.attn_rank)
        self.keys = tuple(self.shapes[0])
        n_all = sum(int(np.prod(s[k])) for s in self.shapes for k in self.keys)
        self.work_flat = torch.empty(n_all, dtype=torch.float32).share_memory_()
        self.work_layers = TW.work_views(self.work_flat, self.shapes)
        init = B.make_lora("cpu", seed=cfg.lora_seed, b_std=0.0)
        for d, lo in zip(self.work_layers, init):
            for k in LORA_KEYS:
                d[k].copy_(lo[k].detach())
        if cfg.attn_rank:  # SkyRL: A kaiming-uniform (a = sqrt(5): U(+-1 / sqrt(fan_in))), B zero; own generator
            g = torch.Generator().manual_seed(int(cfg.lora_seed) + 1000)
            for d in self.work_layers:
                for s in ATTN_SLICES:
                    A = d[f"{s}A"]
                    A.copy_((torch.rand(A.shape, generator=g) * 2 - 1) / float(A.shape[1]) ** 0.5)
                    d[f"{s}B"].zero_()
        self.work = [torch.nn.Parameter(d[k], requires_grad=True) for d in self.work_layers for k in self.keys]
        self.opt = fast_sync(MasterAdamW)(self.work, lr=5e-5, weight_decay=0.0, max_grad_norm=1.0)  # masters = the exact init; work = bf16(master)
        v_first = cfg.visits - int(cfg.K)
        visits = [SimpleVisit(v) for v in range(cfg.visits)]
        self.layers = TL.trained_layers(visits, v_first)
        self.layout = ParamLayout(self.shapes, self.layers, cfg.train_keys)
        self.flats = [torch.zeros(self.layout.size, dtype=torch.float32).share_memory_() for _ in cfg.cards]
        ctx = mp.get_context("spawn")
        self.handles = []
        child_env = {**STACKS[cfg.stack], **WORKER_ENV}       # what the workers import ift_long and the engine under (review B1)
        saved = {k: os.environ.get(k) for k in [*child_env, "CUDA_VISIBLE_DEVICES"]}
        try:
            os.environ.update(child_env)
            for c, flat in zip(physical, self.flats):
                os.environ["CUDA_VISIBLE_DEVICES"] = str(c)
                here, there = ctx.Pipe()
                p = ctx.Process(target=TW.worker_main, args=(there, cfg, self.work_flat, flat, self.shapes, self.layers, c), daemon=True)
                p.start()
                there.close()
                self.handles.append(ProcHandle(p, here, cfg.row_timeout_s))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        try:
            for h in self.handles:                            # each worker reports after its engine load and its stack check
                if not h.conn.poll(900):
                    raise TimeoutError("a worker did not finish loading in 900 s")
                kind, val = h.conn.recv()
                if kind != "ok":
                    raise RuntimeError(f"worker failed to start: {val}")
        except Exception:
            self.close()
            raise
        if masters is not None:
            self.set_masters(masters)

    @staticmethod
    def _physical_cards(cards) -> list:
        """`cards` are physical indices. Under a CUDA_VISIBLE_DEVICES the caller may only name cards it lists (review M5)."""
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if vis is not None:
            allowed = [x.strip() for x in vis.split(",") if x.strip()]
            if not all(x.isdigit() for x in allowed):
                raise ValueError(f"CUDA_VISIBLE_DEVICES={vis!r}: only plain card indices are supported")
            bad = [c for c in cards if str(int(c)) not in allowed]
            if bad:
                raise ValueError(f"cards {bad} are not in this process's CUDA_VISIBLE_DEVICES={vis!r} (cards are physical indices)")
        return [int(c) for c in cards]

    # ------------------------------------------------------------------ adapter
    def _refresh(self):
        errs = []

        def one(h):
            try:
                h.call("refresh")
            except Exception as e:
                errs.append(e)

        th = [threading.Thread(target=one, args=(h,), daemon=True) for h in self.handles]   # every worker reads the same shared work values
        for t in th:
            t.start()
        for t in th:
            t.join()
        if errs:
            raise errs[0]

    def masters(self) -> list:
        """The current fp32 masters in the rig layout (copies)."""
        out, i = [], 0
        for _ in self.shapes:
            out.append({k: self.opt.masters[i + j].detach().clone() for j, k in enumerate(self.keys)})
            i += len(self.keys)
        return out

    def set_masters(self, layers: list) -> None:
        for L, d in enumerate(layers):
            miss = [k for k in self.keys if k not in d]
            if miss:
                raise ValueError(f"set_masters: layer {L} lacks {miss[:4]} (this trainer's keys: {len(self.keys)} per layer; an expert-only adapter at attn_rank > 0?)")
        i = 0
        with torch.no_grad():
            for d in layers:
                for k in self.keys:
                    self.opt.masters[i].copy_(torch.as_tensor(np.asarray(d[k]), dtype=torch.float32))
                    i += 1
        self.opt.sync_work()
        self._refresh()

    def export_adapter(self, out_dir: str, dtype: str = "BF16", meta: dict | None = None) -> str:
        """The engine's adapter file. Expert-only: the kit's writer (unchanged). With attention LoRA: t8_adapter's combined PEFT-native file
        (r = attn_rank, lora_alpha = attn_alpha, rank_pattern 16 for the experts; the engine's served form)."""
        layers = [{k: v.numpy() for k, v in d.items()} for d in self.masters()]
        if self.cfg.attn_rank:
            import t8_adapter as TA

            return TA.export_combined(layers, out_dir, self.cfg.attn_rank, self.cfg.attn_alpha, self.B.RANK, dtype, metadata=meta)
        from surogate.inkling_rl import adapter as AD

        return AD.export_adapter(layers, out_dir, self.B.RANK, int(self.B.ALPHA), dtype, metadata=meta)

    # ------------------------------------------------------------------ gates and reads
    def check_zero_adapter(self, rows: list) -> list:
        """HARD GATE, on every worker. A failure writes STOP."""
        if not rows:
            raise ValueError("check_zero_adapter needs at least one row")
        try:
            return [h.call("check_zero_adapter", r) for h in self.handles for r in rows]
        except T8GateError as e:
            write_stop(self.cfg.run_dir, str(e))
            raise

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
    def compute(self, rows: list, version, step_cfg: StepConfig, seeds: str = "gspo", minibatch_index: int = 0) -> StepOut:
        return compute_step(self.handles, self.layout, self.flats, self.shapes, self.cfg, rows, version, step_cfg, seeds, minibatch_index)

    def ppo_batch(self, rows: list, version, lr: float, n_minibatches: int = 1, step_cfg: StepConfig | None = None) -> list:
        """SkyRL's update on one batch: old log-probs recomputed by the trainer at the start weights, then n_minibatches optimizer steps,
        each on a contiguous part of `rows` that never splits a group (row.group_idx). row.advantage must already hold the GRPO advantage
        (t8_skyrl.grpo_advantages). Returns [(StepOut, grad norm)] per mini-batch."""
        return ppo_batch(self, rows, version, lr, n_minibatches, step_cfg)

    def apply(self, out: StepOut, lr: float) -> float:
        """The kit's fp32 AdamW on the masters. AdamW descends, so only a "descent" gradient is accepted (the sign cannot slip at the hand-over)."""
        if out.sense != "descent":
            raise ValueError('apply() takes a "descent" gradient (seeds="gspo"); for seeds="weights" make the update and call set_masters')
        for g in self.opt.opt.param_groups:
            g["lr"] = lr
        for p, g in zip(self.work, out.grads):
            p.grad = g
        t0 = time.time()
        norm = self.opt.step()
        t1 = time.time()
        self._refresh()
        if TIMING:
            print("T8_TIMING apply " + json.dumps({"optimizer": round(t1 - t0, 3), "refresh": round(time.time() - t1, 3)}), flush=True)
        return norm

    def close(self) -> None:
        for h in self.handles:
            try:
                h.close()
            except Exception:
                pass
        self.handles = []


def split_minibatches(rows: list, n: int) -> list:
    """n contiguous parts of the row indices, cut only where row.group_idx changes (SkyRL cuts by prompts); each cut is the group start nearest to
    an even split in rows that still leaves one group start for every later cut."""
    if n < 1:
        raise ValueError(n)
    starts = [0] + [i for i in range(1, len(rows)) if rows[i].group_idx != rows[i - 1].group_idx]
    if n > len(starts):
        raise ValueError(f"{n} mini-batches for {len(starts)} groups")
    cut_idx = [0]                                             # indices into `starts`
    for j in range(1, n):
        lo, hi = cut_idx[-1] + 1, len(starts) - (n - j)
        target = j * len(rows) / n
        cut_idx.append(min(range(lo, hi + 1), key=lambda k: (abs(starts[k] - target), k)))
    bounds = [starts[k] for k in cut_idx] + [len(rows)]
    return [list(range(bounds[j], bounds[j + 1])) for j in range(n)]


def check_groups_contiguous(rows: list) -> None:
    """Every row carries a group_idx, and each group's rows form ONE contiguous run (SkyRL's mini-batches are prompt-contiguous)."""
    seen, prev = set(), object()
    for r in rows:
        g = r.group_idx
        if g is None:
            raise ValueError("ppo_batch: a row without group_idx")
        if g != prev:
            if g in seen:
                raise ValueError(f"ppo_batch: the rows of group {g} are not contiguous; order the rows by group first")
            seen.add(g)
            prev = g


def ppo_batch(tr, rows: list, version, lr: float, n_minibatches: int, step_cfg: StepConfig | None):
    """All or nothing: on any exception the masters and the optimizer state are put back as they were at entry (review m3)."""
    import copy

    from dataclasses import replace

    step_cfg = step_cfg or StepConfig(max_policy_lag=None)
    if step_cfg.aggregation != "token":
        raise ValueError("token mean")
    if tr.cfg.head != "plain":
        raise ValueError("seeds='ppo' needs head='plain'")
    check_groups_contiguous(rows)
    cap = tr.cfg.max_row_tokens
    need = [r.loss_mask.any() and (not cap or r.T <= cap) for r in rows]  # refused rows get no old log-probs (they would be refused anyway)
    masters0 = [m.detach().clone() for m in tr.opt.masters]
    state0 = copy.deepcopy(tr.opt.opt.state_dict())
    try:
        olds = tr.logprobs([r for r, n in zip(rows, need) if n])   # the start weights; the replay gate reads them in compute
        it = iter(olds)
        for r, n in zip(rows, need):
            r.notes["old_lp"] = next(it) if n else np.zeros(r.T - 1, dtype=np.float32)
        out = []
        for k, part in enumerate(split_minibatches(rows, n_minibatches)):
            o = tr.compute([rows[i] for i in part], version, replace(step_cfg), seeds="ppo", minibatch_index=k)
            out.append((o, tr.apply(o, lr)))
        return out
    except BaseException:
        with torch.no_grad():
            for m, m0 in zip(tr.opt.masters, masters0):
                m.copy_(m0)
        tr.opt.opt.load_state_dict(state0)
        tr.opt.sync_work()
        if hasattr(tr, "_refresh"):
            tr._refresh()
        raise


class SimpleVisit:
    """Physical layer of a visit, without loading the engine (OPS.md section 1)."""

    def __init__(self, v: int):
        self.layer = v if v < 6 else (6 + (v - 6) % 12 if v < 42 else v - 24)
