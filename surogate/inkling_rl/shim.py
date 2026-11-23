"""The Inkling trainer behind the kit's files (the kit's trainer contract, surogate/grpo/trainer.py 87ffd39f + patches).

    python -m surogate.inkling_rl.shim CONFIG.yaml          the trainer process (the kit's run_trainer.sh starts it)
    python -m surogate.inkling_rl.shim CONFIG.yaml --zero-adapter DIR
                                                            export the initial adapter (B = 0) = the run's version 0
CONFIG = the kit's train yaml (output_dir, save_steps, max_steps, learning_rate, max_grad_norm, weight_decay, lora_rank,
lora_alpha, max_policy_lag, loss {...}, surogate_metrics_path) + a section
    inkling_rl: {backend: fake | rig, run_dir: <output_dir>/run_eff, engine_release_sha: <libsinfer sha256 the rollouts
                 were sampled with>, poll_s: 2, wait_timeout_s: 0, broadcast_keep: 4, seed: 0, rig: {...}}

Loop (trainer.py:1130-1240): orch_step = start_step + packs; if orch_step > 0: broadcast step_<orch_step> (adapter files,
then STABLE) and prune old broadcasts; stop at max_steps; wait for <run_dir>/rollouts/step_<orch_step>/rollouts.bin; one
step with version = orch_step; log + metrics; checkpoint when step > 0 and step % save_steps == 0; step += 1. Resume:
start_step = newest checkpoint K + 1 (its weights are broadcast again as step_<K+1>).
Engine identity: the run's engine release sha (libsinfer.so) is recorded in every metrics line and
checkpoint; a backend that loads engine code (the engine-op trainer) must report the same sha, else the shim refuses to start; a batch dir
carrying ENGINE_SHA with another value is refused.
Optimizer: fp32 AdamW on fp32 MASTERS (host), global grad-norm clip; the backend's work parameters hold bf16(master) (the
values a bf16 broadcast hands the engine; with alpha / r = 2 the engine's fold of them is exact), gradients of the work
values are the masters' gradients (straight-through the rounding).
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import time

import numpy as np

from . import adapter as AD
from .records import load_rows
from .step import StepConfig, run_step

STEP_DIR = re.compile(r"^step_(\d+)$")


def bucharest() -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("Europe/Bucharest")).strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{bucharest()}] {msg}", flush=True)


class MasterAdamW:
    """fp32 AdamW on host masters; work params (the backend's) get bf16(master) after every step."""

    def __init__(self, work: list, lr: float, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0,
                 max_grad_norm: float = 1.0):
        import torch

        self.torch = torch
        self.work = list(work)
        self.masters = [p.detach().to("cpu", torch.float32, copy=True) for p in self.work]
        for m in self.masters:
            m.requires_grad_(True)
        self.opt = torch.optim.AdamW(self.masters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.max_grad_norm = max_grad_norm
        self.sync_work()

    def n_params(self) -> int:
        return int(sum(m.numel() for m in self.masters))

    def sync_work(self) -> None:
        torch = self.torch
        with torch.no_grad():
            for p, m in zip(self.work, self.masters):
                r = torch.from_numpy(AD.round_bf16(m.detach().numpy()).reshape(tuple(m.shape)))
                p.copy_(r.to(p.device, p.dtype))

    def step(self) -> float:
        torch = self.torch
        grads = [(p.grad.detach().to("cpu", torch.float32) if p.grad is not None else torch.zeros_like(m))
                 for p, m in zip(self.work, self.masters)]
        norm = math.sqrt(sum(float(g.double().pow(2).sum()) for g in grads))
        coef = min(1.0, self.max_grad_norm / (norm + 1e-6)) if self.max_grad_norm else 1.0
        for m, g in zip(self.masters, grads):
            m.grad = g * coef if coef < 1.0 else g
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        for p in self.work:
            p.grad = None
        self.sync_work()
        return norm

    def state_arrays(self) -> dict:
        out = {}
        for i, m in enumerate(self.masters):
            st = self.opt.state.get(m, {})
            if st:
                out[f"{i}.exp_avg"] = st["exp_avg"].numpy()
                out[f"{i}.exp_avg_sq"] = st["exp_avg_sq"].numpy()
                out[f"{i}.step"] = np.asarray([float(st["step"])], np.float32)
        return out

    def load_state_arrays(self, arrs: dict) -> None:
        torch = self.torch
        for i, m in enumerate(self.masters):
            if f"{i}.exp_avg" in arrs:
                self.opt.state[m] = {"step": torch.tensor(float(arrs[f"{i}.step"][0])),
                                     "exp_avg": torch.from_numpy(np.array(arrs[f"{i}.exp_avg"])),
                                     "exp_avg_sq": torch.from_numpy(np.array(arrs[f"{i}.exp_avg_sq"]))}


def masters_layers(backend, opt: MasterAdamW) -> list:
    """The fp32 masters in the rig layout (per physical layer), in the backend's parameter order."""
    out, i = [], 0
    for d in backend.lora_layers():
        if d is None:
            out.append(None)
            continue
        out.append({k: opt.masters[i + j].detach().numpy() for j, k in enumerate(AD.KEYS)})
        i += len(AD.KEYS)
    return out


def make_backend(cfg: dict):
    ir = cfg.get("inkling_rl") or {}
    kind = ir.get("backend", "fake")
    if kind == "fake":
        from .backend_fake import FakeInkling

        f = ir.get("fake") or {}
        return FakeInkling(seed=int(f.get("seed", 0)), rank=int(cfg.get("lora_rank", 4)), alpha=int(cfg.get("lora_alpha", 8)))
    if kind == "rig":
        from .backend_rig import RigBackend

        return RigBackend(cfg)
    if kind == "engine":
        # the engine-op trainer (surogate/train/inkling_fastpath/backend_engine.py, class
        # EngineBackend(cfg)): the forward = the pinned release libsinfer.so's own ops. Its modules import each other flat,
        # so its directory goes on sys.path (inkling_rl.engine.code_dir; nothing of it is vendored here).
        ec = ir.get("engine") or {}
        code_dir = ec.get("code_dir")
        if not code_dir or not os.path.isdir(code_dir):
            raise ValueError("inkling_rl.engine.code_dir must name a checkout's surogate/train/inkling_fastpath")
        sys.path.insert(0, code_dir)
        from backend_engine import EngineBackend  # noqa: E402

        return EngineBackend(cfg)
    raise ValueError(f"inkling_rl.backend {kind!r}")


class Shim:
    def __init__(self, cfg: dict, backend=None):
        self.cfg = cfg
        ir = cfg.get("inkling_rl") or {}
        self.out = cfg["output_dir"]
        self.run_dir = ir.get("run_dir") or os.path.join(self.out, "run_eff")
        self.metrics_path = cfg.get("surogate_metrics_path") or os.path.join(self.out, "metrics.jsonl")
        self.save_steps = int(cfg.get("save_steps", 3))
        self.max_steps = int(cfg.get("max_steps", 0))
        self.keep = int(ir.get("broadcast_keep", 4))
        self.ckpt_keep = int(ir.get("checkpoint_keep", 3))
        self.poll = float(ir.get("poll_s", 2))
        self.wait_timeout = float(ir.get("wait_timeout_s", 0))
        self.rank, self.alpha = int(cfg.get("lora_rank", 16)), int(cfg.get("lora_alpha", 32))
        self.engine_sha = ir.get("engine_release_sha")
        self.require_sha = bool(ir.get("require_engine_sha", True))
        self.step_cfg = StepConfig.from_train_yaml(cfg)
        self.lr = float(cfg.get("learning_rate", 5e-5))
        if cfg.get("optimizer", "adamw") != "adamw":
            raise ValueError("RL needs fp32 AdamW (optimizer: adamw)")
        self.backend = backend or make_backend(cfg)
        bsha = getattr(self.backend, "engine_sha", None)
        if bsha is not None and self.engine_sha is not None and bsha != self.engine_sha:
            raise SystemExit(f"REFUSED: the trainer backend runs engine code {bsha}, the rollouts' engine release is "
                             f"{self.engine_sha}")
        self.opt = MasterAdamW(self.backend.parameters(), lr=self.lr, weight_decay=float(cfg.get("weight_decay", 0.0)),
                               max_grad_norm=float(cfg.get("max_grad_norm", 1.0)))
        os.makedirs(self.out, exist_ok=True)
        os.makedirs(os.path.dirname(self.metrics_path) or ".", exist_ok=True)
        self.start_step = self._resume()
        log(f"[LoRA] Full-precision AdamW optimizer: {len(self.opt.masters)} tensors, {self.opt.n_params()} params, "
            f"{self.opt.n_params() * 8 / 1e6:.1f} MB state (FP32 m+v)")
        log(f"start inkling_rl shim: backend={type(self.backend).__name__} run_dir={self.run_dir} start_step="
            f"{self.start_step} engine_release_sha={self.engine_sha} backend_engine_sha={bsha}")

    # ------------------------------------------------------------------ files
    def _ckpt_dirs(self) -> list[tuple[int, str]]:
        out = []
        for name in os.listdir(self.out) if os.path.isdir(self.out) else []:
            m = STEP_DIR.match(name)
            if m and len(name) == 13 and os.path.exists(os.path.join(self.out, name, "checkpoint.json")):
                out.append((int(m.group(1)), os.path.join(self.out, name)))
        return sorted(out)

    def _resume(self) -> int:
        ck = [c for c in self._ckpt_dirs() if c[0] > 0]
        if not ck or not self.cfg.get("resume_from_checkpoint", True):
            return 0
        K, path = ck[-1]
        meta = json.load(open(os.path.join(path, "checkpoint.json")))
        if self.engine_sha is not None and meta.get("engine_release_sha") not in (None, self.engine_sha):
            raise SystemExit(f"REFUSED resume: checkpoint {path} was trained on engine release "
                             f"{meta.get('engine_release_sha')}, this run names {self.engine_sha}")
        layers, _ = AD.import_adapter(os.path.join(path, "adapter"), physical_layers=len(self.backend.lora_layers()))
        i = 0
        for d in layers:
            if d is None:
                continue
            for k in AD.KEYS:
                self.opt.masters[i].data.copy_(self.opt.torch.from_numpy(np.ascontiguousarray(d[k])))
                i += 1
        st = AD.read_safetensors(os.path.join(path, "lora_optimizer.rank0.safetensors"))
        self.opt.load_state_arrays({k: v for k, (_, v) in st.items()})
        self.opt.sync_work()
        if "torch_rng" in meta:
            self.opt.torch.set_rng_state(self.opt.torch.frombuffer(bytearray.fromhex(meta["torch_rng"]), dtype=self.opt.torch.uint8))
        log(f"resumed from {path} (step {K}); next batch step_{K + 1}")
        return K + 1

    def export(self, out_dir: str, dtype: str = "BF16", meta: dict | None = None) -> str:
        return AD.export_adapter(masters_layers(self.backend, self.opt), out_dir, self.rank, self.alpha, dtype,
                                 metadata=meta)

    def broadcast(self, n: int) -> None:
        d = os.path.join(self.run_dir, "broadcasts", f"step_{n}")
        os.makedirs(d, exist_ok=True)
        sha = self.export(d, "BF16", meta={"step": n, "engine_release_sha": self.engine_sha})
        with open(os.path.join(d, "STABLE"), "w") as f:
            f.write(sha + "\n")
        steps = sorted(int(m.group(1)) for x in os.listdir(os.path.dirname(d)) if (m := STEP_DIR.match(x)))
        for old in steps[:-self.keep] if len(steps) > self.keep else []:
            shutil.rmtree(os.path.join(os.path.dirname(d), f"step_{old}"), ignore_errors=True)
        log(f"broadcast step_{n} sha256 {sha[:16]}")

    def checkpoint(self, step: int) -> None:
        final = os.path.join(self.out, f"step_{step:08d}")
        tmp = final + ".part"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        sha = self.export(os.path.join(tmp, "adapter"), "F32")
        arrs = self.opt.state_arrays()
        AD.write_safetensors(os.path.join(tmp, "lora_optimizer.rank0.safetensors"),
                             {k: ("F32", np.ascontiguousarray(v, dtype=np.float32)) for k, v in arrs.items()})
        meta = {"step": step, "world_size": 1, "adapter_sha256": sha, "engine_release_sha": self.engine_sha,
                "backend": type(self.backend).__name__, "torch_rng": bytes(self.opt.torch.get_rng_state().numpy()).hex()}
        json.dump(meta, open(os.path.join(tmp, "checkpoint.json"), "w"))
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
        ck = [c for c in self._ckpt_dirs() if c[0] > 0]
        for k, path in ck[:-self.ckpt_keep] if len(ck) > self.ckpt_keep else []:
            shutil.rmtree(path, ignore_errors=True)
        log(f"Saving checkpoint at step {step}... done ({final})")

    def noise_dump(self, step: int) -> None:
        """The gated start's S6 dump (the kit's run/grpo_train_noise_dump.py contract): with env NOISE_DUMP_DIR, after each optimizer step in
        NOISE_DUMP_STEPS (default "0 1") write <dir>/step_<k>/lora_optimizer.rank0.safetensors holding `lora_adamw.state1` = the fp32
        AdamW first moment m of every master, concatenated in the optimizer's order (per layer A13, B13, A2, B2, each flattened in C
        order), then a DONE marker. Outside the run's checkpoint dir; a step already DONE is not dumped again (a trainer restart).
        tools/si/si_noise_check.py recovers g0 = m0 / (1 - b1), g1 = (m1 - b1 m0) / (1 - b1), given Inkling's geometry as arguments
        (plan [24], mods gate_up / down): its defaults are W's and refuse this layout (verdict GEOMETRY)."""
        d = os.environ.get("NOISE_DUMP_DIR")
        if not d or step not in {int(x) for x in os.environ.get("NOISE_DUMP_STEPS", "0 1").split()}:
            return
        out = os.path.join(d, f"step_{step:08d}")
        done = os.path.join(out, "DONE")
        if os.path.exists(done):
            return
        os.makedirs(out, exist_ok=True)
        parts = []
        for master in self.opt.masters:
            st = self.opt.opt.state.get(master) or {}
            parts.append(st["exp_avg"].numpy().ravel() if "exp_avg" in st else np.zeros(master.numel(), np.float32))
        AD.write_safetensors(os.path.join(out, "lora_optimizer.rank0.safetensors"),
                             {"lora_adamw.state1": ("F32", np.concatenate(parts).astype(np.float32))},
                             metadata={"step": step, "engine_release_sha": self.engine_sha})
        with open(done, "w") as f:
            f.write(f"noise dump after optimizer step {step}\n")
        log(f"noise dump after optimizer step {step}: {out}")

    def wait_batch(self, n: int) -> str | None:
        path = os.path.join(self.run_dir, "rollouts", f"step_{n}", "rollouts.bin")
        t0 = time.time()
        while not os.path.exists(path):
            if os.path.exists(os.path.join(self.run_dir, "STOP_TRAINER")):
                return None
            if self.wait_timeout and time.time() - t0 > self.wait_timeout:
                return None
            time.sleep(self.poll)
        shaf = os.path.join(os.path.dirname(path), "ENGINE_SHA")
        if self.engine_sha is not None:
            if not os.path.exists(shaf):
                if self.require_sha:
                    raise SystemExit(f"REFUSED batch step_{n}: no ENGINE_SHA beside it (the kit branch inkling-rl writes one)")
            elif open(shaf).read().strip() != self.engine_sha:
                raise SystemExit(f"REFUSED batch step_{n}: sampled on engine release {open(shaf).read().strip()}, the run "
                                 f"names {self.engine_sha}")
        return path

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        step = self.start_step
        packs = 0
        while True:
            orch = self.start_step + packs
            if orch > 0:
                self.broadcast(orch)
            if 0 < self.max_steps <= orch:
                log(f"Reached max steps ({self.max_steps}), stopping")
                break
            path = self.wait_batch(orch)
            if path is None:
                log(f"no batch step_{orch} (stop file or timeout); stopping")
                break
            packs += 1
            t0 = time.time()
            n_batch, rows = load_rows(path)
            res = run_step(self.backend, rows, orch, self.step_cfg, log=log)
            norm = self.opt.step()
            dt = time.time() - t0
            m = res.metrics
            tokens = int(sum(r.T for r in rows))
            log(f"dispatch-PP step {step}: {res.rows} rows, {tokens} tokens  (inkling_rl shim, backend "
                f"{type(self.backend).__name__}; not dispatch-PP: the kit's gate parses this line)")
            log(f"step={step} loss={m.get('policy_loss', 0):.4f} grad_norm={norm:.4f} lr={self.lr:.2e} "
                f"kl={m.get('mismatch_kl', 0):.4f} masked={m.get('is_masked', 0):.2%} tokens={m.get('total_tokens', 0)} "
                f"micro_batches={res.rows} expected={m.get('expected_loss_scale', 0):g} time={dt:.2f}s")
            entry = {"step": step, "ts": time.time(), "train/policy_loss": m.get("policy_loss", 0.0),
                     "train/mismatch_kl": m.get("mismatch_kl", 0.0), "train/is_masked": m.get("is_masked", 0.0),
                     "train/total_tokens": float(m.get("total_tokens", 0)), "train/grad_norm": norm, "train/lr": self.lr,
                     "train/micro_batches": float(res.rows), "train/expected_loss_scale": m.get("expected_loss_scale", 0.0),
                     "train/duration_ms": dt * 1000.0, "train/orch_step": float(orch),
                     "train/seq_log_ratio": m.get("seq_log_ratio", 0.0),
                     "engine_release_sha": self.engine_sha, "batch_step": n_batch}
            for k, v in m.items():
                if k.startswith(("offpolicy_", "unversioned_", "engine_", "r3_", "refused_")):
                    entry[f"train/{k}"] = float(v)
            with open(self.metrics_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            self.noise_dump(step)
            if self.save_steps > 0 and step > 0 and step % self.save_steps == 0:
                self.checkpoint(step)
            step += 1


def load_config(path: str) -> dict:
    import yaml

    return yaml.safe_load(open(path))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = load_config(argv[0])
    if "--zero-adapter" in argv:
        out = argv[argv.index("--zero-adapter") + 1]
        backend = make_backend(cfg)
        opt = MasterAdamW(backend.parameters(), lr=0.0)
        layers = masters_layers(backend, opt)
        if any(d is not None and (np.any(d["B13"]) or np.any(d["B2"])) for d in layers):
            raise SystemExit("the initial adapter has B != 0")
        sha = AD.export_adapter(layers, out, int(cfg.get("lora_rank", 16)), int(cfg.get("lora_alpha", 32)), "BF16")
        print(sha)
        return 0
    Shim(cfg).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
