"""Opt-in GPU regression against prepared artifacts supplied by the caller.

SUROGATE_TEST_OFFLOAD_MODELS accepts comma-separated paths or glob patterns.
Set CUDA_VISIBLE_DEVICES to the GPU reserved for this test. Fixtures and real
checkpoints remain owned by the caller; this test never changes or deletes them.
"""

import glob
import os
import re
import subprocess
from pathlib import Path

import pytest

from surogate.cli.serve import _resolve_binary
from surogate.serve.artifact.container import Artifact


def artifacts():
    return sorted({Path(path) for pattern in os.environ.get("SUROGATE_TEST_OFFLOAD_MODELS", "").split(",")
                   if pattern for path in glob.glob(pattern)})


@pytest.mark.parametrize("model", artifacts() or [None], ids=lambda p: p.stem if p else "opt-in")
def test_generation_with_host_offload(model):
    if model is None:
        pytest.skip("set SUROGATE_TEST_OFFLOAD_MODELS to prepared test artifacts")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.skip("reserve a test GPU through CUDA_VISIBLE_DEVICES")
    binary = _resolve_binary("generate")
    if binary is None:
        pytest.skip("build the native serving engine first")
    with Artifact(model) as artifact:
        experts = int(artifact.geometry.get("experts", 0))

    def run(flags):
        command = [binary, str(model), "--max-context", "128", "--prefill-chunk", "128",
                   "--max-new", "4", "--greedy", "--prompt", "abc", "--print-token-ids", *flags]
        result = subprocess.run(command, capture_output=True, timeout=240)
        diagnostics = result.stderr.decode(errors="replace")
        assert result.returncode == 0, diagnostics
        ids = re.search(r"tokens\s+generated ids\s+([\d ]+)", diagnostics)
        assert ids is not None, diagnostics
        return ids.group(1).split(), diagnostics

    resident, _ = run(["--gpu-layers", "all"])
    offload_flags = ["--gpu-layers", "0"]
    if experts:
        offload_flags += ["--expert-slots", str(experts)]
    offloaded, _ = run(offload_flags)
    run(["--gpu-layers", "1", *(["--expert-slots", str(experts)] if experts else [])])
    # Exact token parity is useful for fixed synthetic fixtures. Real quantized checkpoints
    # can choose different tokens after host requantization, particularly at close logits.
    if os.environ.get("SUROGATE_TEST_OFFLOAD_EXACT") == "1":
        assert offloaded == resident
    if experts:
        _, diagnostics = run(["--host-moe-layers", "all", "--expert-slots", str(experts),
                              "--cpu-moe-share", "1", "--cpu-moe-prefill-share", "1",
                              "--cpu-moe-min-tokens", "1"])
        assert "CPU expert split enabled" in diagnostics
        run(["--host-moe-layers", "1", "--expert-slots", str(experts),
             "--host-expert-bank", "q4"])
