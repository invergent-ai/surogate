"""CPU test of the shim's S6 noise dump against the KIT'S OWN checker (tools/si/si_noise_check.py, loaded read-only from KIT_RECIPE).
Run: KIT_RECIPE=<kit recipe dir> python tests/inkling_rl/test_noise_dump.py

 1 no NOISE_DUMP_DIR: nothing is written
 2 two steps with the dump on: exactly step_0 and step_1 (a third step is not dumped); the checker's geometry passes and it sees
   lora_A's step-0 gradient as exactly 0 (the A / B segmentation is right)
 3 g0 = m0 / (1 - b1) and g1 = (m1 - b1 m0) / (1 - b1) from the dumps == the gradients the optimizer really applied
 4 the checker's norm over all layers == its own recomputation from those applied gradients
 5 with the kit's W defaults the same dump is refused (verdict GEOMETRY), so W's constants can never judge Inkling silently
 6 a step already DONE is not rewritten (a trainer restart)
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_glue as TG  # noqa: E402

from surogate.inkling_rl import adapter as AD  # noqa: E402
from surogate.inkling_rl.backend_fake import FakeInkling  # noqa: E402
from surogate.inkling_rl.shim import MasterAdamW, Shim, masters_layers  # noqa: E402

B1 = 0.9


def checker():
    path = os.path.join(TG.KIT, "tools", "si", "si_noise_check.py")
    spec = importlib.util.spec_from_file_location("si_noise_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def setup(tmp, max_steps):
    es = TG.kit_loop()
    cfg = TG._cfg(tmp, max_steps=max_steps)
    eng = FakeInkling(seed=0, rank=4, alpha=8)
    t0 = FakeInkling(seed=0, rank=4, alpha=8)
    v0 = os.path.join(tmp, "v0")
    AD.export_adapter(masters_layers(t0, MasterAdamW(t0.parameters(), lr=0.0)), v0, 4, 8, "BF16")
    run = cfg["inkling_rl"]["run_dir"]
    for k in range(max_steps):
        data, _ = TG.make_batch(es, eng, v0, 0, k, [[1, 2, 3], [4, 5], [6, 7, 8, 9]], n=6, new=6, seed=1 + k)
        TG.write_batch(run, k, data)
    return cfg, run


def run_shim(cfg, applied=None):
    sh = Shim(cfg, backend=FakeInkling(seed=0, rank=4, alpha=8))
    if applied is not None:
        inner = sh.opt.opt.step

        def step(*a, **kw):
            applied.append(np.concatenate([m.grad.numpy().ravel() for m in sh.opt.masters]))
            return inner(*a, **kw)

        sh.opt.opt.step = step
    with contextlib.redirect_stdout(io.StringIO()):
        sh.run()


def main():
    tmp = tempfile.mkdtemp()
    old = {k: os.environ.pop(k, None) for k in ("NOISE_DUMP_DIR", "NOISE_DUMP_STEPS")}
    try:
        # 1
        cfg, _ = setup(os.path.join(tmp, "off"), 1)
        run_shim(cfg)
        assert not os.path.exists(os.path.join(tmp, "off", "train.noise"))
        # 2
        cfg, run = setup(os.path.join(tmp, "on"), 3)
        dump = os.path.join(tmp, "on", "train.noise")
        os.environ["NOISE_DUMP_DIR"] = dump
        applied = []
        run_shim(cfg, applied)
        assert sorted(os.listdir(dump)) == ["step_00000000", "step_00000001"], os.listdir(dump)
        assert len(applied) == 3
        E, r, H, I = 8, 4, 16, 8
        mods = [["gate_up", E * r * H, E * 2 * I * r], ["down", E * r * I, E * H * r]]
        per_layer = sum(a + b for _, a, b in mods)
        args = {"dump_dir": dump, "batches": [os.path.join(run, "rollouts", f"step_{k}", "rollouts.bin") for k in (0, 1)],
                "plan": [3], "mods": mods, "groups": {"L0-2": [0, 2]}, "rho_row": {"L0-2": 0.01}}
        ck = checker()
        out = ck.check(args)
        assert out["verdict"] in ("PASS", "FAIL") and out["lora_A_nonzero_step0"] == 0, out
        # 3
        m0 = ck.state1(os.path.join(dump, "step_00000000", "lora_optimizer.rank0.safetensors"))
        m1 = ck.state1(os.path.join(dump, "step_00000001", "lora_optimizer.rank0.safetensors"))
        assert m0.shape[0] == 3 * per_layer == applied[0].shape[0]
        g0, g1 = np.asarray(m0, np.float64) / (1 - B1), (np.asarray(m1, np.float64) - B1 * np.asarray(m0, np.float64)) / (1 - B1)
        for got, ref in ((g0, applied[0]), (g1, applied[1])):
            ref = ref.astype(np.float64)
            assert np.linalg.norm(ref) > 0 and np.linalg.norm(got - ref) / np.linalg.norm(ref) < 2e-5, np.linalg.norm(got - ref)
        # 4 lora_B only, as the checker sums it
        idx, off = [], 0
        for _ in range(3):
            for _, na, nb in mods:
                idx.append(np.arange(off + na, off + na + nb))
                off += na + nb
        idx = np.concatenate(idx)
        row = out["groups"]["L0-2"]
        assert math.isclose(row["norm_g0"], float(np.linalg.norm(g0[idx])), rel_tol=1e-9), row
        assert math.isclose(row["norm_g1"], float(np.linalg.norm(g1[idx])), rel_tol=1e-9), row
        # 5
        assert ck.check({"dump_dir": dump, "batches": args["batches"], "plan": [3]})["verdict"] == "GEOMETRY"
        # 6
        f = os.path.join(dump, "step_00000000", "lora_optimizer.rank0.safetensors")
        before = (os.stat(f).st_mtime_ns, AD.sha256_file(f))
        shutil.rmtree(os.path.join(tmp, "on", "train"))
        shutil.rmtree(os.path.join(run, "broadcasts"))
        run_shim(cfg)
        assert (os.stat(f).st_mtime_ns, AD.sha256_file(f)) == before
        print(f"test_noise_dump: 6 / 6 PASS (checker verdict {out['verdict']}, A nonzero at step 0: {out['lora_A_nonzero_step0']}, "
              f"cos(g0, g1) {row['cos_s0_s1']:.3f})")
    finally:
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
