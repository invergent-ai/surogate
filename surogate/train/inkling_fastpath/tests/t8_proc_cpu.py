"""The step's process machinery on the fake model: two spawned workers == two inline workers bitwise (shared sums, segment tensors over the pipe);
a gate failure in a worker arrives as T8GateError; a dead worker ends the step with an error, it does not hang."""
import copy
import os
import signal
import sys
import tempfile
import threading

import torch
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from surogate.inkling_rl.backend_fake import FakeInkling  # noqa: E402
from surogate.inkling_rl.step import StepConfig  # noqa: E402
import t8_step as TS  # noqa: E402
import t8_step_cpu as C  # noqa: E402


def fake_main(conn, cfg, flat, shapes):
    layout = TS.ParamLayout(shapes, list(range(len(shapes))), cfg.train_keys)
    conn.send(("ok", {"pid": os.getpid()}))
    TS.serve(conn, C.FakeWorker(cfg, flat, layout))


def start(n, cfg):
    be = FakeInkling(seed=0, rank=4, alpha=8)
    shapes = [{k: tuple(d[k].shape) for k in TS.LORA_KEYS} for d in be.lora]
    layout = TS.ParamLayout(shapes, list(range(len(shapes))), cfg.train_keys)
    flats = [torch.zeros(layout.size, dtype=torch.float32).share_memory_() for _ in range(n)]
    ctx = mp.get_context("spawn")
    handles, pids = [], []
    for f in flats:
        here, there = ctx.Pipe()
        p = ctx.Process(target=fake_main, args=(there, cfg, f, shapes), daemon=True)
        p.start()
        there.close()
        kind, val = here.recv()
        assert kind == "ok"
        pids.append(val["pid"])
        handles.append(TS.ProcHandle(p, here))
    return handles, layout, flats, shapes, pids


def main():
    version = 3
    rows = C.make_rows(10, 2, version)
    scfg = StepConfig(max_policy_lag=1)
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    cfg = TS.T8Config(head="plain", max_row_tokens=10 ** 6, segment_keys=("B13", "B2"), run_dir=tmp, visits=5, top_k=2, K=5)
    h, lay, fl, sh, _ = C.trainer_parts(2, cfg)
    ref = TS.compute_step(h, lay, fl, sh, cfg, copy.deepcopy(rows), version, scfg, "gspo")
    h, lay, fl, sh, pids = start(2, cfg)
    try:
        got = TS.compute_step(h, lay, fl, sh, cfg, copy.deepcopy(rows), version, scfg, "gspo")
        assert all(torch.equal(a, b) for a, b in zip(got.grads, ref.grads)), "spawned != inline"
        assert set(got.segments) == set(ref.segments) and all(torch.equal(got.segments[k], ref.segments[k]) for k in ref.segments), "segments"
        print("two spawned workers == two inline workers, bitwise: PASS")
        r = copy.deepcopy(rows)
        t = int(r[3].loss_mask.nonzero()[0][0])
        r[3].engine_lp[t] += 0.5
        try:
            TS.compute_step(h, lay, fl, sh, cfg, r, version, scfg, "gspo")
            raise AssertionError("no gate error")
        except TS.T8GateError as e:
            assert "REPLAY GATE" in str(e)
        print("gate error across the pipe: PASS")
        again = TS.compute_step(h, lay, fl, sh, cfg, copy.deepcopy(rows), version, scfg, "gspo")
        assert all(torch.equal(a, b) for a, b in zip(again.grads, ref.grads)), "a step after a failed step must start from zero"
        print("a step after a failed step: PASS")
        cfg0 = TS.T8Config(head="plain", max_row_tokens=10 ** 6, segment_keys=(), run_dir=tmp, visits=5, top_k=2, K=5)
        h0, lay0, fl0, sh0, _ = start(2, cfg0)
        try:
            o0 = TS.compute_step(h0, lay0, fl0, sh0, cfg0, copy.deepcopy(rows), version, scfg, "gspo")
            assert o0.segments == {} and all(torch.equal(a, b) for a, b in zip(o0.grads, ref.grads)), "no segment keys"
        finally:
            for x in h0:
                x.close()
        print("no segment sums over the pipe: PASS")
        os.kill(pids[1], signal.SIGKILL)                       # this test's own child
        res = {}

        def run():
            try:
                TS.compute_step(h, lay, fl, sh, cfg, copy.deepcopy(rows), version, scfg, "gspo")
                res["r"] = "returned"
            except Exception as e:
                res["r"] = type(e).__name__

        th = threading.Thread(target=run, daemon=True)
        th.start()
        th.join(60)
        assert not th.is_alive(), "the step hangs on a dead worker"
        assert res["r"] != "returned", "a step with a dead worker returned a gradient"
        print(f"dead worker: the step ended with {res['r']}: PASS")
    finally:
        for x in h:
            try:
                x.close()
            except Exception:
                pass
    print("T8 PROC CPU TEST: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
