"""CPU test: accumulating the row gradients on the host (P4 AMENDMENT 3, memory only) changes nothing numerically. The same kit batch goes through the
package at OLD_ROOT (d9839e9d, gradients accumulate in p.grad) and this tree, on the fake backend, two steps, in separate processes. PASS iff the broadcast
adapters step_1 and step_2 are byte-identical and every metric except the wall time is equal.
Run: OLD_ROOT=<dir holding inkling_rl at d9839e9d> KIT_RECIPE=<recipe> python tests/inkling_rl/test_host_grads.py"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_glue as TG  # noqa: E402

from surogate.inkling_rl import adapter as AD  # noqa: E402
from surogate.inkling_rl.backend_fake import FakeInkling  # noqa: E402
from surogate.inkling_rl.shim import MasterAdamW, masters_layers  # noqa: E402

RUN = """
import contextlib, io, json, sys
sys.path.insert(0, sys.argv[1])
from surogate.inkling_rl.backend_fake import FakeInkling
from surogate.inkling_rl.shim import Shim
import surogate.inkling_rl.shim as M
cfg = json.load(open(sys.argv[2]))
with contextlib.redirect_stdout(io.StringIO()):
    Shim(cfg, backend=FakeInkling(seed=0, rank=4, alpha=8)).run()
print(M.__file__)
"""


def run(root, tmp, tag, batches):
    cfg = TG._cfg(os.path.join(tmp, tag), max_steps=2)
    for k, data in enumerate(batches):
        TG.write_batch(cfg["inkling_rl"]["run_dir"], k, data)
    f = os.path.join(tmp, tag + ".json")
    json.dump(cfg, open(f, "w"))
    pp = os.environ.get("PYTHONPATH", "")
    out = subprocess.run([sys.executable, "-c", RUN, root, f], env={**os.environ, "PYTHONPATH": pp}, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-1500:]
    assert out.stdout.strip().startswith(root), (root, out.stdout)
    b = os.path.join(cfg["inkling_rl"]["run_dir"], "broadcasts")
    shas = {k: AD.sha256_file(os.path.join(b, f"step_{k}", "adapter_model.safetensors")) for k in (1, 2)}
    drop = {"ts", "train/duration_ms"}
    met = [{k: v for k, v in json.loads(x).items() if k not in drop} for x in open(cfg["surogate_metrics_path"])]
    return shas, met


def main():
    old = os.environ["OLD_ROOT"]
    new = os.path.dirname(os.path.dirname(HERE))
    tmp = tempfile.mkdtemp()
    es, eng, t0 = TG.kit_loop(), FakeInkling(seed=0, rank=4, alpha=8), FakeInkling(seed=0, rank=4, alpha=8)
    v0 = os.path.join(tmp, "v0")
    AD.export_adapter(masters_layers(t0, MasterAdamW(t0.parameters(), lr=0.0)), v0, 4, 8, "BF16")
    batches = [TG.make_batch(es, eng, v0, 0, k, [[1, 2, 3], [4, 5], [6, 7, 8, 9]], n=6, new=6, seed=1 + k)[0] for k in range(2)]
    a, ma = run(old, tmp, "old", batches)
    b, mb = run(new, tmp, "new", batches)
    assert a == b, (a, b)
    assert ma == mb, [k for k in ma[0] if ma[0][k] != mb[0].get(k)]
    print(f"test_host_grads: PASS (step_1 {a[1][:16]}, step_2 {a[2][:16]} byte-identical; {len(ma[0])} metrics equal over 2 steps, 18 rows a step)")


if __name__ == "__main__":
    main()
